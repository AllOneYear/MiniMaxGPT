import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import bitsandbytes as bnb

from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import tiktoken

# ---------------------------
# 1) CONFIGURATION
# ---------------------------
logger = get_logger(__name__)

OUTPUT_DIR = "./checkpoints_best"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
ACCUMULATION_STEPS = 128
LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.03
MIXED_PRECISION = "fp16"
BLOCK_SIZE = 64
PAD_TOKEN_ID = 200012
WEIGHT_DECAY = 1e-2

# Phase settings: each "phase" is a dict with:
#   - 'name': a label for logging
#   - 'data_dir': which shards directory
#   - 'num_epochs': how many epochs to run
TRAIN_PHASES = [
    {
        "name": "Pretraining",
        "data_dir": "./pretraining_shards",  # e.g. your pretraining shards"./tokenized_shards"
        "num_epochs": 1
    },
    {
        "name": "Finetuning",
        "data_dir": "./finetuning_shards",  # e.g. your O1 shards ./10k_o1_cleaned_tokens
        "num_epochs": 1
    },
]

# Set random seed
#set_seed(SEED)

# Special tokens
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
PAD_TOKEN = "<|pad|>"
THINKING_TOKEN = "<thinking>"
OUTPUT_TOKEN = "<output>"
THOUGHT_TOKEN = "<Thought>"

special_tokens_dict = {
    USER_TOKEN: 200010,
    ASSISTANT_TOKEN: 200011,
    PAD_TOKEN: 200012,
    THINKING_TOKEN: 200013,
    OUTPUT_TOKEN: 200014,
    THOUGHT_TOKEN: 200015,
}

# Tiktoken initialization
base_enc = tiktoken.encoding_for_model("gpt-4o")
encoding = tiktoken.Encoding(
    name="gpt-4o-custom",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **special_tokens_dict},
)
pad_token_id = special_tokens_dict[PAD_TOKEN]

# Define SAVE_EVERY_EPOCH here
SAVE_EVERY_EPOCH = True  # or False, depending on your preference
# ---------------------------
# 2) DATASET CLASS
# ---------------------------
class TokenizedShardsDataset(Dataset):
    """Loads pre-tokenized shards and prepares data for training."""
    def __init__(self, shards_dir, block_size, pad_id):
        self.block_size = block_size
        self.pad_id = pad_id
        self.examples = []

        # List all .txt or .txt.gz files in the directory
        self.files = [
            os.path.join(shards_dir, f)
            for f in os.listdir(shards_dir)
            if f.endswith(".txt") or f.endswith(".txt.gz")
        ]
        if not self.files:
            logger.warning(f"No valid .txt or .txt.gz files found in directory: {shards_dir}")
            return

        logger.info(f"Loading tokenized shards from: {shards_dir}")
        for file in self.files:
            open_func = gzip.open if file.endswith(".gz") else open
            with open_func(file, "rt", encoding="utf-8") as f:
                for line in f:
                    tokens = list(map(int, line.strip().split()))
                    if not tokens:
                        continue  # Skip empty lines
                    for i in range(0, len(tokens), block_size):
                        block = tokens[i : i + block_size]
                        if len(block) < block_size:
                            block += [pad_id] * (block_size - len(block))
                        self.examples.append(block)

        if not self.examples:
            logger.warning(f"No valid tokenized data found in files from: {shards_dir}")
        else:
            logger.info(f"Loaded {len(self.examples)} examples from {len(self.files)} shards.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        block = self.examples[idx]
        input_ids = torch.tensor(block, dtype=torch.long)
        return input_ids, input_ids.clone()

# ---------------------------
# 3) COLLATE FUNCTION
# ---------------------------
def collate_fn(batch):
    input_ids = [b[0] for b in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_masks = (padded_input_ids != PAD_TOKEN_ID).long()
    return padded_input_ids, attention_masks


# ---------------------------
# 4) TRAINING FUNCTION
# ---------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, accelerator, epoch, num_epochs, phase_name=""):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"{phase_name} Epoch {epoch + 1}/{num_epochs}",
                        disable=not accelerator.is_local_main_process)

    for step, (input_ids, attention_masks) in enumerate(progress_bar):
      # Move inputs to the correct device
        input_ids = input_ids.to(accelerator.device)
        attention_masks = attention_masks.to(accelerator.device)
        # Forward pass
        logits, loss = model(input_ids, attention_mask=attention_masks, targets=input_ids)
        loss = loss.mean()  # gradient_accum can produce multiple losses

        accelerator.backward(loss)

        if (step + 1) % accelerator.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        if accelerator.is_local_main_process:
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = epoch_loss / len(dataloader)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    accelerator.print(f"{phase_name} Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}, PPL: {ppl:.4f}")
    return avg_loss, ppl


# ---------------------------
# 5) MAIN SCRIPT
# ---------------------------

def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
        log_with="tensorboard",
        project_dir=os.path.join(OUTPUT_DIR, "logs"),
    )
    logger.info("Accelerator initialized.")

    """config = MiniMaxConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=encoding.n_vocab,
        block_size=64,          # or 1024, etc.
        pad_token_id=pad_token_id,
        dropout=0.1,
        lightning_ratio=7,
        num_experts=4,
        use_post_layernorm=True,
        lightning_block_size=128
    )
    """
    config = MiniMaxConfig(
      vocab_size=encoding.n_vocab,
      block_size=64,
      n_layer=2,
      n_head=2,
      n_embd=128,
      dropout=0.1,
      tie_word_embeddings=True,
      adaptive_xpos=True,
      use_sparse_attn=False,
      use_hybrid_attn=True,
      lightning_ratio=3,
      use_moe=True,
      num_experts=4,
      moe_top_k=2,
      moe_capacity_factor=1.2,
      moe_balance_factor=0.1,
      diversity_factor=0.01,
      use_adaptive_router=False
    )

    model = EnhancedMiniMaxGPT(config)

    #model = MiniMaxGPT(config)

    #model = MiniMaxGPT(mini_config)
    logger.info("Model initialized.")

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    """from lion_pytorch import Lion

    optimizer = Lion(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    from transformers import Adafactor

    optimizer = Adafactor(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, scale_parameter=False, relative_step=False)"""

    # Multi-phase training
    for phase_idx, phase_info in enumerate(TRAIN_PHASES):
        phase_name = phase_info["name"]
        shards_dir = phase_info["data_dir"]
        num_epochs = phase_info["num_epochs"]

        logger.info(f"Starting Phase {phase_idx + 1}: {phase_name}")

        # Create dataset/dataloader
        dataset = TokenizedShardsDataset(shards_dir, BLOCK_SIZE, PAD_TOKEN_ID)
        if len(dataset) == 0:
            logger.error(f"Dataset for {phase_name} is empty. Skipping this phase.")
            continue

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # LR scheduler
        total_steps = (len(dataloader) * num_epochs) // ACCUMULATION_STEPS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

        # Training loop
        for epoch in range(num_epochs):
            train_one_epoch(
                model, dataloader, optimizer, scheduler,
                accelerator, epoch, num_epochs, phase_name=phase_name,
            )
            if SAVE_EVERY_EPOCH and accelerator.is_local_main_process:
                ckpt_path = os.path.join(OUTPUT_DIR, f"{phase_name}_epoch_{epoch + 1}.pth")
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

        if accelerator.is_local_main_process:
            final_ckpt_path = os.path.join(OUTPUT_DIR, f"{phase_name}_final.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), final_ckpt_path)
            logger.info(f"Saved final checkpoint for {phase_name}.")

    if accelerator.is_local_main_process:
        final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
        torch.save(accelerator.unwrap_model(model).state_dict(), final_model_path)
        logger.info(f"Training complete. Final model saved at {final_model_path}")

if __name__ == "__main__":
    main()
