"""
Fine-Tuning Flan-T5
======================================
This script fine-tunes the instruction-tuned T5 model 'google/flan-t5-small'
on your custom text data to map natural language instructions to structured commands.

What this does:
1. Loads a pre-trained Flan-T5 model
2. Loads your text files from the 'data/' folder
3. Fine-tunes the model on input→output pairs
4. Saves the trained model to 'agent-trained/' folder

Requirements:
- Text files in 'data/' folder (e.g., data/example.txt)
- Text format: 
  Each line should be: "### Input: <instruction> ### Output: <structured commands>"
- At least 4GB of available memory recommended
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW 
from datasets import load_dataset
from tqdm import tqdm
import os

def main():
    """Main training function"""

    # ============================
    # STEP 1: Load model & tokenizer
    # ============================
    print("📦 Loading Flan-T5 model and tokenizer...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Ensure model uses pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # ============================
    # STEP 2: Load dataset
    # ============================
    print("📂 Loading dataset from data/*.txt...")
    dataset = load_dataset("text", data_files={"train": "data/*.txt"})["train"]
    print(f"   Found {len(dataset)} text samples")

    # ============================
    # STEP 3: Tokenize data
    # ============================
    def preprocess(examples):
        """
        Tokenize the input/output text.
        Assumes each line is: "### Input: ... ### Output: ..."
        """
        # Split input/output
        texts = examples["text"]
        inputs = []
        targets = []

        for text in texts:
            # Split on "### Output:" to separate input from output
            parts = text.split("### Output:")
            if len(parts) == 2:
                # Input is everything before "### Output:"
                inputs.append(parts[0].strip())
                # Target is everything after "### Output:"
                targets.append(parts[1].strip())
            else:
                # Fallback if format is unexpected
                inputs.append(text)
                targets.append("")

        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            truncation=True,
            padding="max_length",
            max_length=128
        )
        return model_inputs

    print("🔢 Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])

    # Convert to PyTorch tensors
    tokenized_dataset.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'labels']
    )

    # ============================
    # STEP 4: DataLoader
    # ============================
    train_dataloader = DataLoader(
        tokenized_dataset, batch_size=4, shuffle=True, num_workers=0
    )

    # ============================
    # STEP 5: Optimizer
    # ============================
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # ============================
    # STEP 6: Train loop
    # ============================
    NUM_EPOCHS = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    print(f"🚀 Training on {device} for {NUM_EPOCHS} epochs...\n")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_dataloader)
        print(f"✅ Epoch {epoch+1} finished. Average loss: {avg_loss:.4f}")

    # ============================
    # STEP 7: Save model
    # ============================
    save_dir = "./agent-trained"
    print(f"\n💾 Saving fine-tuned model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✨ Training complete!")

if __name__ == "__main__":
    main()
