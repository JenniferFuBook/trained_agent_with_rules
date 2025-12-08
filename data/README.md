# 📁 Training Data Folder

This folder contains the text files used to train your GPT-2 model.

## 📋 What Goes Here

Put your `.txt` files in this folder. The training script will automatically:
- Find all `.txt` files in this folder
- Read their contents
- Use them to train the model

## ✅ File Requirements

### File Format
- **Extension**: Must be `.txt` (not `.TXT`, `.text`, or other)
- **Encoding**: UTF-8 (standard text encoding)
- **Content**: Plain text only (no Word docs, PDFs, or images)

### File Naming
```bash
# ✅ Good file names:
training_data.txt
shakespeare.txt
customer_support_logs.txt
my_writing_samples.txt

# ❌ Avoid:
data.TXT          # Wrong extension case
file.doc          # Wrong format
data with spaces.txt  # Spaces can cause issues (use underscores)
```

## 📝 Content Guidelines

### Good Training Data

```txt
# ✅ GOOD: Clear, consistent, well-formatted text

Example 1 - Narrative Writing:
Once upon a time in a distant kingdom, there lived a brave knight.
The knight embarked on many adventures across the land.
Each journey taught valuable lessons about courage and wisdom.

Example 2 - Technical Documentation:
To install the package, run the following command.
The installation process takes approximately 5 minutes.
After installation, verify the setup by running the test suite.

Example 3 - Conversational:
Customer: How do I reset my password?
Agent: You can reset your password by clicking the "Forgot Password" link.
Customer: Where can I find that link?
Agent: It's located on the login page, below the password field.
```

### Poor Training Data

```txt
# ❌ BAD: Inconsistent formatting, mixed styles

aSDFklj random text!!! @#$%
normal sentence here.
ALL CAPS SHOUTING TEXT
🎉🎉🎉 too many emojis 🎉🎉🎉
<html><body>HTML code</body></html>
```

## 📊 Data Quality Tips

### 1. Volume
- **Minimum**: 1,000 words (about 2-3 pages)
- **Recommended**: 10,000+ words (about 20 pages)
- **Optimal**: 50,000+ words (about 100 pages)

More data = Better results (generally)

### 2. Consistency
Keep writing style consistent within each file:
- Same tone (formal vs casual)
- Same domain (technical vs creative)
- Same format (prose vs dialogue)

### 3. Quality Over Quantity
```bash
# ✅ Better: 1,000 words of high-quality, relevant text
# ❌ Worse: 10,000 words of random, low-quality text
```

### 4. Clean Text
Remove or minimize:
- Excessive special characters (!@#$%^&*)
- HTML/XML tags (<div>, </p>, etc.)
- Code snippets (unless training a code model)
- Multiple consecutive blank lines
- Non-ASCII characters (if possible)

## 📂 Organization Strategies

### Option 1: Single File
Best for: Simple projects, homogeneous data

```
data/
└── my_training_data.txt  (all your text in one file)
```

### Option 2: Multiple Files by Topic
Best for: Diverse content, organized datasets

```
data/
├── fiction_stories.txt
├── technical_docs.txt
├── conversational_data.txt
└── product_descriptions.txt
```

### Option 3: Multiple Files by Source
Best for: Aggregating data from different sources

```
data/
├── source1_website.txt
├── source2_books.txt
├── source3_articles.txt
└── source4_my_writing.txt
```

## 🎯 Example Training Data Files

### Example 1: Creative Writing

**File**: `creative_writing.txt`
```txt
The old lighthouse stood alone on the rocky cliff, its beacon cutting through the fog.
Waves crashed against the shore below, their rhythm constant and eternal.
Inside, the keeper tended the light, ensuring ships would find their way home.

Years passed, seasons changed, but the lighthouse remained steadfast.
Storms tested its strength, but it never wavered.
It became a symbol of hope for all who saw its light.
```

### Example 2: Technical Documentation

**File**: `technical_docs.txt`
```txt
System Installation Guide

Step 1: Download the installer from the official website.
Ensure you have administrator privileges before proceeding.

Step 2: Run the installer executable.
Follow the on-screen prompts to select your installation directory.

Step 3: Configure the initial settings.
Choose your preferred language and timezone during setup.

Step 4: Complete the installation.
The system will automatically restart after installation finishes.
```

### Example 3: Customer Support

**File**: `support_conversations.txt`
```txt
Customer: I'm having trouble logging into my account.
Agent: I'd be happy to help. Can you tell me what error message you're seeing?

Customer: It says "Invalid credentials" even though I'm using the right password.
Agent: Let's try resetting your password. I'll send a reset link to your email.

Customer: Thank you, I received the email and reset my password.
Agent: Great! Please try logging in now with your new password.

Customer: It worked! I'm logged in successfully now.
Agent: Wonderful! Is there anything else I can help you with today?
```

## ⚠️ What to Avoid

### 1. Copyrighted Material
```txt
❌ Don't use:
- Published books (without permission)
- News articles (without license)
- Song lyrics
- Movie scripts
```

### 2. Sensitive Information
```txt
❌ Never include:
- Personal information (names, addresses, SSN)
- Passwords or API keys
- Credit card numbers
- Private communications
```

### 3. Harmful Content
```txt
❌ Avoid:
- Hate speech
- Explicit content (unless specifically required)
- Misinformation
- Spam or gibberish
```

## 🔍 Checking Your Data

Before training, verify your data:

```bash
# Count words in your data
wc -w data/*.txt

# View first 10 lines of each file
head -n 10 data/*.txt

# Check file encoding
file data/*.txt

# Search for potential issues
grep -r '[^\x00-\x7F]' data/  # Find non-ASCII characters
```

## 📈 Data Preparation Checklist

Before running training:

- [ ] All files are `.txt` format
- [ ] Files contain at least 1,000 words total
- [ ] Text is clean and well-formatted
- [ ] No copyrighted or sensitive information
- [ ] Consistent writing style within files
- [ ] Files are UTF-8 encoded
- [ ] No HTML/XML tags (unless intentional)
- [ ] Minimal special characters
- [ ] Text is relevant to your use case

## 💡 Pro Tips

### Tip 1: Test with Small Dataset First
Start with a small dataset (100-500 words) to:
- Test your setup
- Verify the training process works
- Iterate quickly

### Tip 2: Use Line Breaks Strategically
```txt
# ✅ Good: Natural paragraph breaks
This is the first paragraph with related content.
It talks about topic A.

This is the second paragraph.
It introduces topic B.

# ❌ Bad: Random line breaks
This sentence is
broken up randomly for
no good reason.
```

### Tip 3: Balance Your Dataset
If training for multiple purposes, balance the amount of each type:
```txt
30% customer support conversations
30% technical documentation
40% product descriptions
```

### Tip 4: Deduplicate Content
Remove duplicate sentences or paragraphs to avoid bias:
```bash
# Find duplicates
sort data/*.txt | uniq -d
```

## 🚀 Quick Start Examples

### Create Sample Data for Testing

```bash
# Navigate to data folder
cd data

# Create a simple test file
cat > test_data.txt << 'EOF'
This is example training data for testing.
The model will learn patterns from this text.
You can add as many sentences as you want.
Make sure the content is relevant to your use case.
Training works best with clean, consistent text.
EOF

# Verify the file was created
cat test_data.txt
```

## 📚 Need More Data?

If you need more training data:

1. **Write Your Own**: Most authentic and relevant
2. **Public Datasets**: Hugging Face Datasets, Common Crawl
3. **Open Source**: Project Gutenberg (out-of-copyright books)
4. **Generate**: Use existing models to create synthetic data
5. **Scrape**: Legally scrape public websites (with permission)

---

**Ready to Train?**

Once your data is prepared, go back to the main folder and run:
```bash
# Using Docker
docker-compose up

# Or using shell script
./run_train.sh

# Or direct Python
python3 train.py
```

Good luck with your training! 🎉
