# 🤖 GPT-2 Training Guide for Beginners

A complete, beginner-friendly guide to fine-tuning GPT-2 on your own text data.

## 📖 Table of Contents

- [What is This?](#what-is-this)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Understanding the Files](#understanding-the-files)
- [Configuration Guide](#configuration-guide)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## 🎯 What is This?

This project helps you **fine-tune** a GPT-2 language model on your own text data. Fine-tuning means teaching an existing AI model to write in a specific style or about specific topics by training it on your custom text.

### What You Can Do With a Fine-Tuned Model:

- 📝 Generate text in a specific writing style
- 💬 Create chatbots with domain-specific knowledge
- ✍️ Auto-complete text based on your data
- 🎨 Generate creative content (stories, poems, etc.)

### Example Use Cases:

- Train on Shakespeare's works to generate Shakespearean text
- Train on customer support tickets to build a support bot
- Train on code repositories to create a code assistant
- Train on your own writing to create a personal writing assistant

---

## 📋 Requirements

### Minimum Requirements:

- **Python**: 3.10 or higher
- **RAM**: At least 2GB available
- **Storage**: 2GB free space (for models and data)
- **OS**: macOS, Linux, or Windows (with WSL)

### Optional (Recommended):

- **Docker**: For easier setup and avoiding system conflicts
- **GPU**: Not required, but speeds up training significantly

---

## 📦 Installation

### Option 1: Using Docker (Recommended for Beginners)

```bash
# 1. Install Docker Desktop from https://www.docker.com/products/docker-desktop

# 2. Verify Docker is installed
docker --version

# That's it! Docker handles all Python dependencies automatically.
```

### Option 2: Manual Python Installation

```bash
# 1. Verify Python version (must be 3.10+)
python3 --version

# 2. Install required packages
pip install -r requirements.txt

# 3. Verify installation
python3 -c "import torch, transformers; print('✅ Installation successful!')"
```

---

## 📁 Project Structure

```
agent/
│
├── 📄 train.py              # Main training script (fully commented)
├── 📄 run_train.sh          # Shell launcher for macOS/Linux
│
├── 🐳 Dockerfile            # Docker container definition
├── 🐳 docker-compose.yml    # Docker orchestration config
├── 📋 requirements.txt      # Python package dependencies
│
├── 📁 data/                 # ⭐ PUT YOUR .txt FILES HERE
│   └── example.txt          # Your training data
│
├── 📁 agent-gpt2/           # Trained model saved here (auto-created)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
│
├── 📄 README_TRAINING.md    # This file
└── 📄 .dockerignore         # Files to exclude from Docker
```

---

## 🚀 Quick Start

### Step 1: Prepare Your Training Data

Create text files in the `data/` folder:

```bash
# Create the data folder
mkdir -p data

# Add your training text
cat > data/my_text.txt << 'EOF'
This is example training data.
You can add as much text as you want.
The model will learn patterns from this text.
Add more lines, paragraphs, and examples.
EOF
```

**Tips for good training data:**
- ✅ At least 1000 words (more is better)
- ✅ Consistent style and format
- ✅ Clean, well-formatted text
- ❌ Avoid mixing very different writing styles
- ❌ Remove excessive special characters or formatting

### Step 2: Choose Your Training Method

#### Method A: Docker (Recommended) ⭐

Best for: Beginners, avoiding system issues, consistent results

```bash
# Build the Docker image (first time only)
docker-compose build

# Run training
docker-compose up

# View logs if running in background
docker-compose logs -f
```

#### Method B: Shell Script (macOS/Linux)

Best for: Direct control, faster iteration

```bash
# Make script executable (first time only)
chmod +x run_train.sh

# Run training
./run_train.sh
```

#### Method C: Direct Python Execution

Best for: Debugging, customization

```bash
# Run directly
python3 train.py
```

### Step 3: Monitor Training Progress

You'll see output like this:

```
📦 Loading model and tokenizer...
📂 Loading your text data...
   Found 150 text samples
🔢 Converting text to numbers (tokenizing)...
⚙️  Setting up training configuration...

🚀 Starting training for 2 epochs...

Epoch 1/2: 100%|████████████| 75/75 [02:15<00:00,  1.81s/it, loss=3.2456]
✅ Epoch 1 completed. Average loss: 3.2456

Epoch 2/2: 100%|████████████| 75/75 [02:10<00:00,  1.74s/it, loss=2.8912]
✅ Epoch 2 completed. Average loss: 2.8912

💾 Saving trained model...
✨ Training complete! Model saved to './agent-gpt2/'
```

**What the output means:**
- **Loss**: How "wrong" the model is (lower = better)
- **Progress bar**: Shows current progress through the data
- **Average loss**: Overall performance for that epoch

### Step 4: Use Your Trained Model

After training, test your model:

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load your trained model
model = GPT2LMHeadModel.from_pretrained("./agent-gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("./agent-gpt2")

# Generate text
prompt = "Once upon a time"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

# Print generated text
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📚 Understanding the Files

### 1. `train.py` - Main Training Script

**What it does:**
1. Loads the pre-trained DistilGPT2 model (82M parameters)
2. Reads your text files from `data/` folder
3. Converts text to numbers (tokenization)
4. Trains the model on your data
5. Saves the fine-tuned model to `agent-gpt2/`

**Key sections:**
- Lines 28-40: Load model and tokenizer
- Lines 43-73: Load and prepare data
- Lines 76-106: Configure training settings
- Lines 109-147: Training loop
- Lines 150-156: Save model

### 2. `run_train.sh` - Shell Launcher

**What it does:**
- Sets environment variables to prevent threading issues on macOS
- Runs the Python training script
- Reports success/failure

**Why use it:**
Prevents the "mutex lock failed" error common on macOS by configuring the environment before Python starts.

### 3. `Dockerfile` - Container Definition

**What it does:**
- Defines a clean, isolated environment for training
- Installs Python 3.10 and all dependencies
- Sets up the workspace

**Why use it:**
Ensures consistent behavior across different computers and operating systems.

### 4. `docker-compose.yml` - Container Orchestration

**What it does:**
- Simplifies Docker commands
- Mounts your data and output folders
- Sets memory limits (4GB max)

**Why use it:**
Makes Docker easier to use with simple commands like `docker-compose up`.

### 5. `requirements.txt` - Python Dependencies

**What it includes:**
- `torch`: PyTorch deep learning framework
- `transformers`: Hugging Face transformers library
- `datasets`: Data loading and processing
- `tqdm`: Progress bars

---

## ⚙️ Configuration Guide

You can customize training by editing `train.py`:

### Common Configurations:

#### 1. Model Size

**Location:** Line 35

```python
# Options (in order of size):
model = GPT2LMHeadModel.from_pretrained("distilgpt2")      # 82M params (default)
model = GPT2LMHeadModel.from_pretrained("gpt2")            # 124M params
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")     # 355M params
model = GPT2LMHeadModel.from_pretrained("gpt2-large")      # 774M params
```

**Trade-offs:**
- Smaller models = Less memory, faster training, lower quality
- Larger models = More memory, slower training, higher quality

#### 2. Maximum Text Length

**Location:** Line 59

```python
max_length=256  # Default: 256 tokens (about 200 words)

# Options:
max_length=128   # Shorter sequences, less memory
max_length=512   # Longer sequences, more context, more memory
max_length=1024  # Maximum for GPT-2, requires lots of memory
```

#### 3. Dataset Size Limit

**Location:** Line 67

```python
MAX_SAMPLES = 100  # Default: limit to 100 samples

# Options:
MAX_SAMPLES = 50    # Smaller dataset, faster training
MAX_SAMPLES = 500   # Larger dataset, better results (needs more memory)
# Remove the limit entirely by commenting out lines 68-70
```

#### 4. Batch Size

**Location:** Line 92

```python
batch_size=1  # Default: process 1 sample at a time

# Options:
batch_size=2   # 2x faster, but uses 2x memory
batch_size=4   # 4x faster, but uses 4x memory
```

#### 5. Learning Rate

**Location:** Line 101

```python
lr=5e-5  # Default: 0.00005

# Options:
lr=1e-5   # Slower learning, more stable
lr=1e-4   # Faster learning, might be unstable
```

#### 6. Number of Epochs

**Location:** Line 105

```python
NUM_EPOCHS = 2  # Default: train for 2 full passes

# Options:
NUM_EPOCHS = 1   # Faster, less thorough training
NUM_EPOCHS = 5   # Slower, more thorough training
```

### Example Configurations:

#### Fast Training (Testing)
```python
MAX_SAMPLES = 20
NUM_EPOCHS = 1
max_length = 128
```

#### Balanced (Default)
```python
MAX_SAMPLES = 100
NUM_EPOCHS = 2
max_length = 256
```

#### High Quality (More Time/Memory)
```python
MAX_SAMPLES = 500
NUM_EPOCHS = 5
max_length = 512
batch_size = 2
model = GPT2LMHeadModel.from_pretrained("gpt2")
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions:

#### 1. "Out of Memory" Error

**Symptoms:**
```
RuntimeError: CUDA out of memory
```
or
```
Killed (exit code 137)
```

**Solutions:**
- ✅ Reduce `MAX_SAMPLES` to 50 or fewer
- ✅ Reduce `max_length` to 128
- ✅ Reduce `batch_size` to 1
- ✅ Use smaller model (`distilgpt2`)
- ✅ Close other applications

#### 2. "Mutex Lock Failed" Error

**Symptoms:**
```
terminating due to uncaught exception of type std::__1::system_error: mutex lock failed
```

**Solutions:**
- ✅ Use Docker: `docker-compose up` (best solution)
- ✅ Use shell script: `./run_train.sh`
- ✅ Downgrade PyTorch: `pip install torch==2.1.0`

#### 3. "Permission Denied" Error

**Symptoms:**
```
bash: ./run_train.sh: Permission denied
```

**Solution:**
```bash
chmod +x run_train.sh
```

#### 4. No Training Data Found

**Symptoms:**
```
FileNotFoundError: data/*.txt
```

**Solutions:**
- ✅ Create `data/` folder: `mkdir data`
- ✅ Add at least one `.txt` file in `data/`
- ✅ Check file extension is `.txt` not `.TXT` or `.text`

#### 5. Training is Very Slow

**Normal:** Training on CPU is slow (2-10 minutes for 100 samples)

**Speed it up:**
- ✅ Reduce `NUM_EPOCHS` to 1
- ✅ Reduce `MAX_SAMPLES` to 50
- ✅ Use GPU if available
- ✅ Use smaller model (`distilgpt2`)

#### 6. Loss is Not Decreasing

**Symptoms:**
Loss stays high (>5.0) or increases

**Solutions:**
- ✅ Reduce learning rate to `1e-5`
- ✅ Check your training data quality
- ✅ Train for more epochs
- ✅ Ensure training data is text-based, not binary

---

## 🎓 Advanced Topics

### Understanding Key Concepts:

#### 1. What is Fine-Tuning?

Fine-tuning takes a pre-trained model and adapts it to your specific data. Think of it like:
- Pre-trained model = A person who speaks English
- Fine-tuning = Teaching them medical terminology

#### 2. What is a Token?

A token is a piece of text (word, part of word, or punctuation). Example:
- "Hello world!" → ["Hello", " world", "!"] = 3 tokens
- "GPT-2" → ["G", "PT", "-", "2"] = 4 tokens

#### 3. What is Loss?

Loss measures how wrong the model's predictions are:
- High loss (>5.0) = Model is very confused
- Medium loss (2.0-4.0) = Model is learning
- Low loss (<2.0) = Model is doing well

#### 4. What is an Epoch?

One epoch = one complete pass through your entire dataset. If you have 100 samples and train for 2 epochs, the model sees all 100 samples twice.

#### 5. What is Gradient Accumulation?

A trick to simulate larger batch sizes without using more memory. Instead of updating after each sample, we accumulate gradients for several samples, then update.

### Using Your Model in Production:

```python
# Save for deployment
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load anywhere
from transformers import pipeline

generator = pipeline('text-generation', model='./my_model')
result = generator("Your prompt here", max_length=50)
print(result[0]['generated_text'])
```

### Monitoring Training with TensorBoard:

```python
# Add to requirements.txt:
# tensorboard>=2.13.0

# In train.py, add logging:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# In training loop:
writer.add_scalar('Loss/train', loss.item(), step)
```

Then view in browser:
```bash
tensorboard --logdir=runs
```

---

## 📖 Additional Resources

### Learning Resources:
- [Hugging Face Course](https://huggingface.co/course) - Free NLP course
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Learn PyTorch
- [Understanding GPT-2](https://jalammar.github.io/illustrated-gpt2/) - Visual guide

### Documentation:
- [Transformers Docs](https://huggingface.co/docs/transformers)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Docker Docs](https://docs.docker.com/)

### Community:
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)

---

## 🤝 Getting Help

If you're stuck:

1. Check the [Troubleshooting](#troubleshooting) section
2. Read error messages carefully
3. Search for the error on Google/Stack Overflow
4. Ask on Hugging Face forums
5. Check if your data is formatted correctly

---

## 📝 License

This project uses:
- PyTorch (BSD License)
- Transformers (Apache 2.0 License)
- GPT-2 models (OpenAI - MIT License)

---

**Happy Training! 🎉**

If you found this helpful, consider sharing it with others learning AI!
