### Enhanced MiniMaxGPT README

---

## Overview

**MiniMaxGPT** is a state-of-the-art transformer-based architecture designed for efficient and scalable natural language processing (NLP) tasks. It integrates cutting-edge innovations in attention mechanisms, memory optimization, sparse computation, and advanced regularization techniques to achieve high performance, training stability, and inference speed. The architecture is highly modular and flexible, making it suitable for a wide range of applications, from large-scale pretraining to specialized fine-tuning tasks.

### Key Features

1. **Gradient Checkpointing**
   - **Purpose**: Reduces GPU memory usage during training by recomputing activations on the fly.
   - **Benefit**: Enables training of larger models and using larger batch sizes on memory-constrained hardware.

2. **Optimized Lightning Attention**
   - **Flash-Attention Integration**: Leverages FlashAttention (when available) for accelerated computations.
   - **Rotary Embeddings**: Incorporates rotary positional embeddings (RoPE) to enhance the model's ability to handle relative positional information.
   - **Hybrid Attention Modes**: Supports configurable hybrid attention mechanisms, allowing for dynamic switching between different attention types (e.g., lightning vs. softmax).
   - **KV Cache Support**: Includes optional KV caching for efficient autoregressive inference.

3. **Memory-Efficient Mixture of Experts (MoE)**
   - **Top-k Gating**: Implements top-k expert selection with capacity limiting to ensure efficient routing of tokens.
   - **Load Balancing**: Enforces balanced expert utilization through auxiliary loss terms.
   - **Diversity Regularization**: Encourages diversity among experts to prevent redundancy and improve generalization.
   - **Vectorized Dispatch**: Utilizes vectorized operations to minimize Python loops, enhancing computational efficiency.

4. **Enhanced Expert Blocks**
   - **Orthogonal Initialization**: Experts are initialized with orthogonal weights to improve training stability and convergence.
   - **Residual Connections**: Incorporates residual connections to facilitate gradient flow and mitigate vanishing gradients.
   - **Layer Scaling**: Implements layer scaling to stabilize training, especially in deep architectures.

5. **Advanced RMSNorm**
   - **Numerical Stability**: Provides a numerically stable alternative to traditional LayerNorm.
   - **FP16 Compatibility**: Safe for use with half-precision (FP16) training and inference, reducing memory usage and accelerating computations.

6. **Rotary Positional Embeddings with Adaptive Scaling**
   - **AdaptiveXPosRotaryEmbedding**: Dynamically scales positional embeddings based on layer depth and sequence length, enhancing long-sequence learning and gradient propagation.

7. **Configurable Hyperparameters**
   - **Flexible Configuration**: The `MiniMaxConfig` dataclass allows for easy customization of model architecture and training parameters, including:
     - Model depth and width (e.g., `n_layer`, `n_embd`)
     - Attention mechanisms (e.g., `use_flash_attn`, `use_hybrid_attn`)
     - MoE settings (e.g., `num_experts`, `moe_top_k`)
     - Normalization strategies (e.g., `use_post_layernorm`)
     - Embedding tying and positional embedding options

### Model Components

1. **Embedding Layer**
   - **Token and Positional Embeddings**: Utilizes `nn.Embedding` layers for token and positional embeddings.
   - **Weight Tying**: Supports optional weight tying between input and output embeddings to reduce the number of parameters.

2. **Transformer Blocks**
   - **EnhancedHybridBlock**: A stack of transformer blocks that combine attention and MoE/MLP layers.
   - **Gradient Checkpointing**: Implements checkpointing to save GPU memory during training.

3. **Attention Mechanism**
   - **Optimized Multi-Head Attention**: Implements multi-head attention with optional rotary embeddings.
   - **Flash-Attention**: Integrates FlashAttention for efficient memory usage during training.
   - **KV Cache**: Supports KV caching for autoregressive inference.

4. **Mixture of Experts (MoE)**
   - **Vectorized Dispatch**: Efficiently routes tokens to experts using vectorized operations.
   - **Balanced Routing**: Ensures balanced expert utilization through auxiliary loss terms.
   - **Diversity Enforcement**: Promotes diversity among experts to enhance model capacity and generalization.

5. **Output Layer**
   - **Linear Projection**: Projects the final representations to the vocabulary size.
   - **Weight Tying**: Optionally ties the output layer weights to the embedding layer to reduce the number of parameters.

### Requirements

#### Dependencies
- **PyTorch 1.12+**: The core deep learning framework.
- **CUDA Toolkit**: Required for GPU training and inference.
- **Optional Libraries**:
  - **Transformers**: For tokenizer support and model interfacing.
  - **tiktoken**: For efficient tokenization.

#### Hardware Recommendations
- **GPU with Sufficient Memory**: Recommended for training large model variants.
- **Memory-Efficient Features**: Gradient checkpointing and other optimizations make it feasible to train on mid-tier GPUs with limited memory.

### Configuration

The model's configuration is managed via the `MiniMaxConfig` dataclass. Below is an example configuration:

```python
config = MiniMaxConfig(
    n_layer=12,            # Number of transformer layers
    n_head=8,              # Number of attention heads
    n_embd=512,            # Embedding dimension
    vocab_size=200000,      # Vocabulary size - my current setup is upto 200,016 but its subject to change
    block_size=1024,       # Maximum sequence length
    dropout=0.1,           # Dropout rate
    use_checkpoint=True,   # Enable gradient checkpointing
    use_moe=True,          # Enable Mixture of Experts
    num_experts=4,         # Number of experts in MoE
    moe_top_k=2            # Top-k gating for MoE
)
```

**Customization Options**:
- **Model Depth and Width**: Adjust `n_layer` and `n_embd` to scale the model.
- **Attention Mechanisms**: Toggle `use_flash_attn` and `use_hybrid_attn` to enable or disable specific attention types.
- **MoE Settings**: Modify `num_experts`, `moe_top_k`, and `moe_capacity_factor` to control MoE behavior.
- **Embedding and Layer Norm**: Enable `adaptive_xpos` and `use_post_layernorm` to enhance positional embeddings and normalization.

### Usage

#### Training

```python
model = EnhancedMiniMaxGPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
targets = torch.randint(0, config.vocab_size, (batch_size, sequence_length))

logits, loss = model(input_ids, targets=targets)
loss.backward()
optimizer.step()
```

#### Inference

```python
model.eval()
input_ids = torch.tensor([[1, 2, 3]])  # Example input
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
```

### Key Hyperparameters

- **Attention Mechanisms**:
  - `use_flash_attn`: Toggles the use of FlashAttention (if available).
  - `use_hybrid_attn`: Enables hybrid attention methods.

- **MoE Routing**:
  - `num_experts`: Number of experts in the MoE layer.
  - `moe_top_k`: Top-k gating for selecting active experts.
  - `moe_capacity_factor`: Controls the capacity per expert.

- **Embedding and Layer Norm**:
  - `adaptive_xpos`: Enables adaptive scaling for rotary embeddings.
  - `use_post_layernorm`: Uses post-layer normalization instead of pre-layer normalization.

### Advantages

- **Memory Efficiency**: Gradient checkpointing and FlashAttention optimize GPU memory usage, enabling training of larger models.
- **Scalability**: MoE and sparse attention enhance performance for large-scale tasks.
- **Flexibility**: Configurable architecture supports both dense and sparse computations.
- **Stability**: Auxiliary losses ensure balanced training and minimize collapse in MoE layers.

### Future Work

- **Sparse Attention**: Extend the architecture to incorporate sparsity-based techniques for extremely long sequences.
- **Adaptive Routing**: Implement adaptive routing to further improve the efficiency of MoE layers.
- **Better Initialization**: Investigate dynamic scaling and advanced initialization techniques for diverse training setups.

### Acknowledgments

This architecture builds upon foundational transformer techniques and incorporates modern optimizations from recent advancements in:

- **FlashAttention**
- **Mixture-of-Experts (MoE)**
- **XPos positional embeddings**

For further inquiries or contributions, feel free to reach out!

### License

Distributed under the MIT License.

---

### Additional Notes

- **Training Stability**: The combination of RMSNorm, orthogonal initialization, and auxiliary losses in the MoE layer contributes to training stability, especially in deep and wide models.
- **Inference Speed**: The use of FlashAttention and KV caching can significantly accelerate inference, making the model suitable for real-time applications.
- **Scalability**: The modular design allows for easy scaling of the model, whether by increasing the number of layers, embedding dimensions, or the number of experts in the MoE layer.

### Example Usage

Below is an example of how to initialize and train the model:

```python
from model import MiniMaxConfig, EnhancedMiniMaxGPT
import torch

# Define configuration
config = MiniMaxConfig(
    n_layer=12,
    n_head=8,
    n_embd=512,
    vocab_size=30000,
    block_size=1024,
    dropout=0.1,
    use_checkpoint=True,
    use_moe=True,
    num_experts=4,
    moe_top_k=2
)

# Initialize model and optimizer
model = EnhancedMiniMaxGPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Example input and targets
input_ids = torch.randint(0, config.vocab_size, (16, 512))
targets = torch.randint(0, config.vocab_size, (16, 512))

# Forward pass
logits, loss = model(input_ids, targets=targets)
loss.backward()
optimizer.step()
```

This example demonstrates the basic training loop. For more advanced training (e.g., multi-phase training, mixed precision), refer to the provided `train.py` script.

---

Feel free to explore and modify the architecture to better suit your specific NLP tasks and requirements. If you have any questions or need further assistance, don't hesitate to ask!
