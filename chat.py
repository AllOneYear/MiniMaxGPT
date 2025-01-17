import os
import torch
import tiktoken

# -----------------------------------------------------------------------------
# 1) SPECIAL TOKENS & TIKTOKEN SETUP
# -----------------------------------------------------------------------------
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

# Initialize the tokenizer
base_enc = tiktoken.encoding_for_model("gpt-4o")
encoding = tiktoken.Encoding(
    name="gpt-4-custom",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **special_tokens_dict},
)
pad_token_id = special_tokens_dict[PAD_TOKEN]

# -----------------------------------------------------------------------------
# 2) IMPORT MINIMAXGPT & CONFIG
# -----------------------------------------------------------------------------
# Assuming you have a file `minimax_model.py` with:
#   from minimax_model import MiniMaxGPT, MiniMaxConfig
#
# If your model code is in the same file, just comment out the import below.
#from minimax_model import MiniMaxGPT, MiniMaxConfig

def get_device():
    """Return GPU device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 3) LOAD THE MODEL CHECKPOINT
# -----------------------------------------------------------------------------
def load_minimax_model(checkpoint_path, config, device):
    """
    Load the MiniMaxGPT model from a .pth checkpoint,
    handling potential size mismatches.
    """
    model = EnhancedMiniMaxGPT(config).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_dict = torch.load(checkpoint_path, map_location=device)

    # Get the state dictionary for the current model
    model_state_dict = model.state_dict()

    # Filter out weights with mismatched shapes
    compatible_state_dict = {}
    for k, v in checkpoint_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            compatible_state_dict[k] = v
        else:
            print(f"Skipping weight {k} due to shape mismatch.")  # Optional: Log skipped weights

    # Load the compatible weights
    model.load_state_dict(compatible_state_dict, strict=False)

    model.eval()
    return model
# -----------------------------------------------------------------------------
# 4) PROMPT / CONVERSATION BUILDING
# -----------------------------------------------------------------------------
def build_prompt(conversation_history):
    """
    Convert a list of dicts [{'role': 'user'/'assistant', 'content': ...}, ...]
    into a text prompt for the model.
    """
    prompt = ""
    for turn in conversation_history:
        if turn["role"] == "user":
            prompt += f"{USER_TOKEN} {turn['content'].strip()}\n"
        else:
            prompt += f"{ASSISTANT_TOKEN} {turn['content'].strip()}\n"
    # End with assistant token to prompt model's next response
    prompt += f"{ASSISTANT_TOKEN} "
    return prompt

def generate_response(
    model,
    encoding,
    conversation_history,
    device,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
):
    """
    Generate a response from the model using the model's generate function.
    """
    prompt_text = build_prompt(conversation_history)

    # Encode the prompt into input IDs
    input_ids = torch.tensor(
        encoding.encode(prompt_text, allowed_special=set(special_tokens_dict.keys())),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    # Ensure the sequence length does not exceed model config
    if input_ids.size(1) > model.config.block_size:
        input_ids = input_ids[:, -model.config.block_size:]

    # Use the model's built-in generate() method
    generated_ids = model.generate(
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Decode only the newly generated tokens
    new_tokens = generated_ids[0].tolist()[len(input_ids[0]):]
    response_text = encoding.decode(new_tokens).strip()
    return response_text

# -----------------------------------------------------------------------------
# 5) MAIN CHAT LOOP
# -----------------------------------------------------------------------------
def main():
    """
    Interactive chat with MiniMaxGPT.
    """
    checkpoint_path = "checkpoints_best/final_model.pth"  # Example path
    device = get_device()
    print(f"Using device: {device}")

    # Build config for MiniMaxGPT
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

    # Load model
    model = load_minimax_model(checkpoint_path, config, device)
    print("MiniMaxGPT loaded successfully.\n")

    # Interactive chat loop
    conversation_history = []
    print("=== MiniMaxGPT Chat ===")
    print("Type 'exit' or 'quit' to end the chat.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break

        # Add user's turn
        conversation_history.append({"role": "user", "content": user_input})

        try:
            # Generate assistant's response
            response = generate_response(
                model=model,
                encoding=encoding,
                conversation_history=conversation_history,
                device=device,
                max_new_tokens=100,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            conversation_history.pop()  # remove the user's turn if failure
            continue

        # Add assistant's turn and print it
        conversation_history.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
