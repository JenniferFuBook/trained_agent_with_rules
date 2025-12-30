import os
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==================================================
# Step 1: Environment setup
# ==================================================
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ==================================================
# Step 2: Load model and tokenizer
# ==================================================
model_name = "./agent-trained"
print(f"🤖 Loading model {model_name}...\n")
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ Model loaded successfully!\n")

# ==================================================
# Step 3: Function to generate structured rules
# ==================================================
def generate_rules(natural_input, max_length=150, temperature=0.7):
    """
    Convert natural language instruction into structured commands:
    ADD_NODE, DELETE_NODE, CONNECT, DISCONNECT
    """
    # Construct a clear instruction prompt matching training format
    # Only include the input part, model will generate the output
    prompt = f"### Input: {natural_input}"
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # Generate up to 100 NEW tokens
        min_new_tokens=1,    # Force at least 1 token
        do_sample=False,      # Greedy decoding for deterministic output
        pad_token_id=tokenizer.pad_token_id,
        num_beams=1
    )

    # Debug: print token-level information
    print(f"🔍 Input token IDs: {inputs['input_ids'][0].tolist()}")
    print(f"🔍 Input tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    print(f"🔍 Output token IDs: {outputs[0].tolist()}")
    print(f"🔍 Output tokens: {tokenizer.convert_ids_to_tokens(outputs[0])}")
    print(f"🔍 Number of output tokens: {len(outputs[0])}")

    # Decode generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Debug: print raw model output
    print(f"🔍 Raw model output: '{text}'")
    print(f"🔍 Output length: {len(text)} characters")

    # Keep only structured commands
    commands = []
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith(("ADD_NODE", "DELETE_NODE", "CONNECT", "DISCONNECT")):
            commands.append(line)
    return "\n".join(commands)

# ==================================================
# Step 4: Main execution
# ==================================================
def main():
    parser = argparse.ArgumentParser(description="Convert natural language to structured commands")
    parser.add_argument("prompt", type=str, help="Natural language instruction to convert")
    args = parser.parse_args()

    print(f"📝 Input: {args.prompt}")
    print("-" * 60)
    result = generate_rules(args.prompt)
    print(f"✨ Output:\n{result}")
    print("-" * 60)

if __name__ == "__main__":
    main()
