"""
Use Your Trained GPT-2 Model
============================
This script loads your fine-tuned model and generates text.

What it does:
1. Sets up environment to prevent errors (macOS)
2. Loads your trained model from agent-gpt2/
3. Takes a prompt and generates text
4. Prints the result

How to use:
- Edit the 'prompt' variable (line 52) to change what you generate
- Run with: python3 use_model.py
- Or safer: ./run_model.sh
- Or Docker: ./docker_use_model.sh
"""

import os
import sys

# ============================================
# STEP 1: Prevent Threading Errors
# ============================================
# IMPORTANT: Set these BEFORE importing libraries
# This prevents "mutex lock failed" error on macOS

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================
# STEP 2: Import Libraries
# ============================================
# Now it's safe to import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def main():
    """Main function - loads model and generates text"""

    # ============================================
    # STEP 3: Load Your Trained Model
    # ============================================
    print("🤖 Loading your trained model...\n")

    # Figure out model path (Docker uses /app, local uses ./)
    model_path = "/app/agent-gpt2" if sys.platform == "linux" else "./agent-gpt2"

    # Load the model and tokenizer
    # local_files_only=True means don't try to download from internet
    model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path, local_files_only=True)

    # Fix: Set padding token to avoid warnings
    # GPT-2 doesn't have a pad token by default, so we use eos_token
    tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model loaded successfully!\n")

    # ============================================
    # STEP 4: Generate Text
    # ============================================

    # 📝 EDIT THIS to change what text is generated
    prompt = "Large language models are"

    print(f"📝 Prompt: {prompt}")
    print("-" * 60)

    # Convert text to numbers (tokens) with proper attention mask
    # return_tensors="pt" means return PyTorch tensors
    # This creates both input_ids and attention_mask automatically
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate new text!
    # Parameters explained:
    # - max_length: How long the output can be (in tokens, ~1 token = 0.75 words)
    # - do_sample: Use randomness (True) vs always pick most likely word (False)
    # - temperature: How creative (0.1=boring, 1.5=wild). 0.7 is balanced
    # - attention_mask: Tells model which tokens to pay attention to
    # - pad_token_id: Technical - tells model what padding looks like
    outputs = model.generate(
        inputs['input_ids'],                     # The tokenized text
        attention_mask=inputs['attention_mask'], # Which tokens are real (not padding)
        max_length=100,                          # Generate up to 100 tokens (~75 words)
        do_sample=True,                          # Use sampling for variety
        temperature=0.7,                         # Balanced creativity
        pad_token_id=tokenizer.pad_token_id      # Use the pad token we set above
    )

    # Convert numbers back to text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ============================================
    # STEP 5: Show Results
    # ============================================
    print(f"✨ Generated:\n{text}\n")
    print("-" * 60)
    print("\n💡 Tips:")
    print("   - Edit line 52 to change the prompt")
    print("   - Increase max_length (line 70) for longer output")
    print("   - Adjust temperature (line 72) for more/less creativity")
    print("   - Run ./docker_use_model.sh if you get errors")


# ============================================
# Run the main function
# ============================================
# This only runs if you execute this file directly
# (not when importing it as a module)
if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError:
        print("\n❌ Error: Model not found!")
        print("📁 Make sure you've trained the model first:")
        print("   docker-compose up")
        print("\n💡 Or check that agent-gpt2/ folder exists with model files")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Try using Docker instead:")
        print("   ./docker_use_model.sh")
