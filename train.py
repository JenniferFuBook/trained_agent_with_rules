"""
Fine-Tuning GPT-2
======================================
This script trains (fine-tunes) a GPT-2 language model on your custom text data.

What this does:
1. Loads a pre-trained GPT-2 model
2. Loads your text files from the 'data/' folder
3. Trains the model to learn patterns from your text
4. Saves the trained model to 'agent-gpt2/' folder

Requirements:
- Text files in 'data/' folder (e.g., data/example.txt)
- At least 2GB of available memory
"""

# Import required libraries
import torch  # PyTorch - the deep learning framework
from torch.utils.data import DataLoader  # Helps load data in batches
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # GPT-2 model and tokenizer
from datasets import load_dataset  # Load and process datasets
from tqdm import tqdm  # Progress bar to see training progress


def main():
    """Main training function - runs when you execute this script"""

    # ============================================
    # STEP 1: Load the Model and Tokenizer
    # ============================================
    print("📦 Loading model and tokenizer...")

    # Load DistilGPT2 - a smaller, faster version of GPT-2
    # 82 million parameters (vs 124M for standard GPT-2)
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")

    # Set padding token (needed for batch processing)
    # We use the end-of-sentence token as the padding token
    tokenizer.pad_token = tokenizer.eos_token


    # ============================================
    # STEP 2: Load and Prepare Your Data
    # ============================================
    print("📂 Loading your text data...")

    # Load all .txt files from the 'data/' folder
    dataset = load_dataset("text", data_files={"train": "data/*.txt"})["train"]
    print(f"   Found {len(dataset)} text samples")

    # Convert text to numbers (tokenization)
    # Models work with numbers, not text, so we convert each word/character to a number
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],           # The text to convert
            truncation=True,            # Cut off text longer than max_length
            padding="max_length",       # Pad shorter text to max_length
            max_length=256              # Maximum 256 tokens per sample (saves memory)
        )

    print("🔢 Converting text to numbers (tokenizing)...")
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Limit dataset size to avoid running out of memory
    # You can increase this if you have more RAM available
    MAX_SAMPLES = 100
    if len(dataset) > MAX_SAMPLES:
        print(f"   Limiting to {MAX_SAMPLES} samples to save memory")
        dataset = dataset.select(range(MAX_SAMPLES))

    # Convert to PyTorch format
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


    # ============================================
    # STEP 3: Configure Training Settings
    # ============================================
    print("⚙️  Setting up training configuration...")

    # Move model to CPU (not GPU) - more compatible, less memory
    device = torch.device('cpu')
    model.to(device)
    model.train()  # Put model in training mode

    # Enable gradient checkpointing - trades speed for memory savings
    model.gradient_checkpointing_enable()

    # Create data loader - feeds data to the model in batches
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,      # Process 1 sample at a time (low memory usage)
        shuffle=True,      # Randomize order each epoch
        num_workers=0      # Don't use multiprocessing (avoids errors on macOS)
    )

    # Create optimizer - adjusts model weights to reduce errors
    # Adam is a popular optimization algorithm
    optimizer = torch.optim.AdamW(
        model.parameters(),  # The model weights to optimize
        lr=5e-5             # Learning rate - how fast the model learns (0.00005)
    )

    # Training configuration
    NUM_EPOCHS = 2              # How many times to go through entire dataset
    ACCUMULATION_STEPS = 2      # Accumulate gradients over 2 steps (simulates batch size of 2)


    # ============================================
    # STEP 4: Training Loop
    # ============================================
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...\n")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0  # Track total error for this epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for step, batch in enumerate(progress_bar):
            # Move data to CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass - model makes predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # For language modeling, labels = inputs
            )

            # Calculate loss (error) - how wrong the model's predictions are
            loss = outputs.loss / ACCUMULATION_STEPS

            # Backward pass - calculate gradients (how to adjust weights)
            loss.backward()

            # Update model weights every ACCUMULATION_STEPS
            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()        # Update weights
                optimizer.zero_grad()   # Reset gradients

            # Track loss for reporting
            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})

        # Print epoch summary
        avg_loss = total_loss / len(train_dataloader)
        print(f"✅ Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")


    # ============================================
    # STEP 5: Save the Trained Model
    # ============================================
    print("\n💾 Saving trained model...")
    model.save_pretrained("./agent-gpt2")
    tokenizer.save_pretrained("./agent-gpt2")
    print("✨ Training complete! Model saved to './agent-gpt2/'")


# This ensures the code only runs when you execute this file directly
# (not when importing it as a module)
if __name__ == '__main__':
    main()

