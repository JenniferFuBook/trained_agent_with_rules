import os
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
tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    # Construct a clear instruction prompt
    prompt = f"""Translate the following instruction into structured commands using only ADD_NODE, DELETE_NODE, CONNECT, DISCONNECT.

Instruction: {natural_input}
"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    prompt = "Add node cache and db, connect them, then delete api."

    print(f"📝 Input: {prompt}")
    print("-" * 60)
    result = generate_rules(prompt)
    print(f"✨ Output:\n{result}")
    print("-" * 60)

if __name__ == "__main__":
    main()
