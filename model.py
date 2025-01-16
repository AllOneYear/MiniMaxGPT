import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
from dataclasses import dataclass


# ------------------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------------------
@dataclass
class MiniMaxConfig:
    """
    Extended config for 'EnhancedMiniMaxGPT' with:
      - gradient checkpointing
      - MoE enhancements
      - advanced attention
      - partial memory, training stability features
    """
    # Basic GPT
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    vocab_size: int = 30000
    block_size: int = 1024
    dropout: float = 0.1
    pad_token_id: int = 0
    bias: bool = False
    tie_word_embeddings: bool = True

    # Memory & training
    use_checkpoint: bool = True
    layer_norm_eps: float = 1e-5
    init_scale: float = 0.02

    # XPos
    rope_base: int = 10000
    rope_scale_base: float = 512.0
    adaptive_xpos: bool = False

    # Attention
    use_hybrid_attn: bool = True
    lightning_ratio: int = 7
    lightning_block_size: int = 256
    use_flash_attn: bool = False
    kv_cache: bool = False

    # MoE
    use_moe: bool = True
    num_experts: int = 4
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.2
    moe_balance_factor: float = 0.1
    diversity_factor: float = 0.01
    expert_dropout: float = 0.1
    z_loss_factor: float = 1e-4  # extra loss factor to stabilize gating

    # LN style
    use_post_layernorm: bool = True

    # Possibly adaptive routing or sparse attn can be added if needed
    use_sparse_attn: bool = False

    use_adaptive_router: bool = False  # Add this line


# ------------------------------------------------------------------------
# 2) Enhanced RMSNorm with FP16 Safety
# ------------------------------------------------------------------------
class EnhancedRMSNorm(nn.Module):
    """
    RMSNorm with better numerical stability (especially in FP16).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # keep track of original dtype
        orig_dtype = x.dtype
        # upcast to float32 for stable mean if in FP16
        if x.dtype == torch.float16:
            x = x.float()
        normed = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        # cast back
        normed = normed.to(orig_dtype)
        return self.weight * normed


# ------------------------------------------------------------------------
# 3) Adaptive XPos Rotary Embedding
# ------------------------------------------------------------------------
class AdaptiveXPosRotaryEmbedding(nn.Module):
    """
    If adaptive_xpos is True, apply an extra scaling factor based on layer depth.
    """
    def __init__(self, dim, base=10000, scale_base=512.0, adaptive=True):
        super().__init__()
        assert dim % 2 == 0, "XPos dimension must be even."
        self.dim = dim
        self.base = base
        self.scale_base = scale_base
        self.adaptive = adaptive

        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, layer_depth=None, dtype=torch.float32):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        scale = self.scale_base ** (t / self.scale_base)
        if self.adaptive and layer_depth is not None:
            scale *= torch.exp(-layer_depth / self.scale_base)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        scaled_freqs = freqs * scale.unsqueeze(-1)
        emb = torch.cat([scaled_freqs, scaled_freqs], dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


def rotate_half(x: torch.Tensor):
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat([-x2, x1], dim=-1)


def apply_xpos_rotary_pos_emb(q, k, cos, sin):
    B, nh, T, hd = q.shape
    cos = cos[:, :, :T, :hd]
    sin = sin[:, :, :T, :hd]

    def rope(x):
        return x * cos + rotate_half(x) * sin

    return rope(q), rope(k)


# ------------------------------------------------------------------------
# 4) OptimizedLightningAttention
# ------------------------------------------------------------------------
class OptimizedLightningAttention(nn.Module):
    """
    'Lightning' multi-head attention with optional flash-attn and kv_cache.
    """
    def __init__(self, config: MiniMaxConfig):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_flash = config.use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        self.kv_cache_enabled = config.kv_cache
        self.register_buffer('kv_cache', None, persistent=False)

        if config.adaptive_xpos:
            self.xpos = AdaptiveXPosRotaryEmbedding(
                dim=self.head_dim,
                base=config.rope_base,
                scale_base=config.rope_scale_base,
                adaptive=config.use_adaptive_router
            )
        else:
            self.xpos = None

    def _shape_heads(self, x: torch.Tensor, B: int, T: int):
        # reshape for [B, n_head, T, head_dim]
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: Optional[int] = None  # For adaptive XPos scaling
    ):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = self._shape_heads(q, B, T)
        k = self._shape_heads(k, B, T)
        v = self._shape_heads(v, B, T)

        # handle past kv if caching
        if layer_past is not None and self.kv_cache_enabled:
            pk, pv = layer_past
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        if self.kv_cache_enabled:
            self.kv_cache = (k, v)

        # Apply Rotary Positional Embedding if enabled
        if self.xpos is not None:
            cos, sin = self.xpos(seq_len=T, device=x.device, layer_depth=layer_idx)
            q, k = apply_xpos_rotary_pos_emb(q, k, cos, sin)

        if self.use_flash:
            # use built-in flash-attn
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = torch.matmul(att, v)

        # merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ------------------------------------------------------------------------
# 5) EnhancedExpertBlock (for MoE experts)
# ------------------------------------------------------------------------
class EnhancedExpertBlock(nn.Module):
    """Expert MLP with better init and optional dropout."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
            nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)

        self.layer_scale = nn.Parameter(torch.ones(1,1,hidden_dim)*0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # simple residual approach for stability
        r = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x * self.layer_scale
        return r + x


# ------------------------------------------------------------------------
# 6) Memory-Efficient MoE with Vectorized Dispatch
# ------------------------------------------------------------------------
class MemoryEfficientMoE(nn.Module):
    """
    MoE with top-k gating, capacity limiting, load balancing, diversity.
    Attempt to reduce Python loops for dispatch.
    """
    def __init__(self, config: MiniMaxConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.capacity_factor = config.moe_capacity_factor
        self.balance_factor = config.moe_balance_factor
        self.diversity_factor = config.diversity_factor
        self.z_loss_factor = config.z_loss_factor
        self.hidden_dim = config.n_embd
        self.dropout = config.expert_dropout

        # build experts
        self.experts = nn.ModuleList([
            EnhancedExpertBlock(self.hidden_dim, self.dropout) for _ in range(self.num_experts)
        ])
        self.router = nn.Linear(self.hidden_dim, self.num_experts)

        # track auxiliary loss
        self.register_buffer('aux_loss', torch.zeros(1))
        self.register_buffer('diversity_loss', torch.zeros(1))

    def compute_diversity_loss(self):
        # measure pairwise cos-sim among experts
        param_vecs = []
        for e in self.experts:
            pvec = []
            for p in e.parameters():
                pvec.append(p.flatten())
            param_vecs.append(torch.cat(pvec, dim=0))

        div_loss = 0.0
        for i in range(self.num_experts):
            for j in range(i+1, self.num_experts):
                cos_sim = F.cosine_similarity(
                    param_vecs[i].unsqueeze(0),
                    param_vecs[j].unsqueeze(0)
                )
                div_loss += cos_sim**2
        return div_loss * self.diversity_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        N = B * T
        E = self.num_experts
        device = x.device

        # gating
        router_logits = self.router(x.view(N, C))  # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)

        # optional z_loss => push router_logits^2 to be small
        z_loss = self.z_loss_factor * (router_logits ** 2).mean()

        # compute importance for load-balance
        importance = router_probs.mean(dim=0)  # (E,)
        target = torch.ones_like(importance) / E
        balance = F.mse_loss(importance, target, reduction='sum') * self.balance_factor

        # pick top-k
        top_vals, top_inds = torch.topk(router_probs, self.top_k, dim=-1)  # (N, K)
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)  # Normalize

        # capacity
        capacity = int(self.capacity_factor * (N // E + 1))

        # Initialize output buffer
        out = torch.zeros_like(x.view(N, C), device=device)

        # Track how many tokens each expert has used
        used_slots = torch.zeros(E, dtype=torch.int32, device=device)

        for i_k in range(self.top_k):
            w = top_vals[:, i_k]        # (N,)
            e_idx = top_inds[:, i_k]    # (N,)

            # Filter out tokens with negligible weights
            mask = w > 1e-9
            if not mask.any():
                continue
            valid_idx = mask.nonzero(as_tuple=True)[0]

            # Iterate over experts
            for eid in range(E):
                # Find tokens for this expert
                mask_eid = (e_idx[valid_idx] == eid)
                count_e = mask_eid.sum().item()
                if count_e == 0:
                    continue
                c_before = used_slots[eid].item()
                c_after = c_before + count_e
                if c_before >= capacity:
                    continue
                if c_after > capacity:
                    allowed = capacity - c_before
                    # Select only allowed tokens
                    selected = mask_eid.nonzero(as_tuple=True)[0][:allowed]
                    real_idx = valid_idx[selected]
                    used_slots[eid] = capacity
                else:
                    selected = mask_eid.nonzero(as_tuple=True)[0]
                    real_idx = valid_idx[selected]
                    used_slots[eid] += count_e

                if len(real_idx) == 0:
                    continue

                # Dispatch tokens to experts
                tokens = x.view(N, C)[real_idx]
                y_ = self.experts[eid](tokens)
                w_ = w[real_idx].unsqueeze(-1)
                out[real_idx] += w_ * y_

        # Sum up auxiliary losses
        self.aux_loss = balance + z_loss
        self.diversity_loss = self.compute_diversity_loss()

        return out.view(B, T, C)


# ------------------------------------------------------------------------
# 7) EnhancedHybridBlock with Checkpointing
# ------------------------------------------------------------------------
class EnhancedHybridBlock(nn.Module):
    """
    Transformer block with optional:
    - 'OptimizedLightningAttention'
    - 'MemoryEfficientMoE' or fallback MLP
    - RMSNorm
    - gradient checkpointing
    """
    def __init__(self, config: MiniMaxConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn = OptimizedLightningAttention(config)

        if config.use_moe:
            self.mlp = MemoryEfficientMoE(config)
        else:
            # fallback
            self.mlp = EnhancedExpertBlock(config.n_embd, config.dropout)

        self.ln_1 = EnhancedRMSNorm(config.n_embd, eps=config.layer_norm_eps)
        self.ln_2 = EnhancedRMSNorm(config.n_embd, eps=config.layer_norm_eps)

        self.use_checkpoint = config.use_checkpoint

    def _forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.config.use_post_layernorm:
            # post-LN
            a_out = self.attn(x, mask, layer_idx=self.layer_idx)
            x = x + a_out
            x = self.ln_1(x)
            m_out = self.mlp(x)
            x = x + m_out
            x = self.ln_2(x)
        else:
            # pre-LN
            a = self.ln_1(x)
            a_out = self.attn(a, mask, layer_idx=self.layer_idx)
            x = x + a_out
            m = self.ln_2(x)
            m_out = self.mlp(m)
            x = x + m_out
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, mask)
        else:
            return self._forward(x, mask)


# ------------------------------------------------------------------------
# 8) EnhancedMiniMaxGPT
# ------------------------------------------------------------------------
class EnhancedMiniMaxGPT(nn.Module):
    """
    Final integrated GPT architecture with:
    - gradient checkpointing
    - advanced MoE
    - optimized lightning attention
    - improved RMSNorm
    - optional XPos or standard
    """
    def __init__(self, config: MiniMaxConfig):
        super().__init__()
        self.config = config

        # embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # build blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layer):
            blk = EnhancedHybridBlock(config, layer_idx)
            self.blocks.append(blk)

        self.ln_f = EnhancedRMSNorm(config.n_embd, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        print(f"[EnhancedMiniMaxGPT] #params (non-embeddings): {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_scale)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
            n_params -= self.wpe.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ):
        B, T = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        if T > self.config.block_size:
            raise ValueError(f"Seq length {T} > block_size {self.config.block_size}")

        pos_ids = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.wte(input_ids) + self.wpe(pos_ids)  # (B, T, C)
        x = self.drop(x)

        # Pass through Transformer blocks
        for layer_idx, blk in enumerate(self.blocks):
            x = blk(x, mask=attention_mask)  # Each block handles layer indexing

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()  # (B, T-1, vocab_size)
            shift_targets = targets[..., 1:].contiguous()  # (B, T-1)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # ((B*(T-1)), vocab_size)
                shift_targets.view(-1),  # (B*(T-1))
                ignore_index=self.config.pad_token_id
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        device = idx.device
        generated = idx

        for _ in range(max_new_tokens):
            idx_cond = generated[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            logits = torch.nan_to_num(logits, nan=float('-inf'))

            if top_k is not None:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                sorted_logits[remove_mask] = float('-inf')
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)  # (B, T+1)

        return generated
