# Hackathon Submission Guide

This guide explains how to set up Ding-Bot for hackathon evaluation.

## Quick Start for Evaluators

### 1. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Node dependencies (for devtools - optional)
cd devtools
npm install
```

### 2. Configure Model Repository

**The model repository is hardcoded in `src/main.py`** (line ~95).

**To set your repository:**
1. Open `src/main.py`
2. Find line: `HUGGINGFACE_MODEL_REPO = 'YOUR_USERNAME/chess-bot-model'`
3. Replace with your repository: `HUGGINGFACE_MODEL_REPO = 'your-username/chess-bot-model'`

**Alternative (for testing):** You can still use environment variable:
```bash
export HUGGINGFACE_MODEL_REPO='your-username/chess-bot-model'
```
This will override the hardcoded value if it's still set to the placeholder.

### 3. Run Server

```bash
# Using startup script (recommended)
./start.sh

# Or directly
python3 serve.py
```

The model will automatically download from Hugging Face on first run.

## How Model Loading Works

1. **Check for local model**: Ding-Bot first checks `models/` directory for a local model file
2. **Check environment variable**: If no local model, checks `HUGGINGFACE_MODEL_REPO` environment variable
3. **Download from Hugging Face**: If set, automatically downloads model from Hugging Face Hub
4. **Cache model**: Downloaded model is cached in `models/` directory for future runs

## For Hackathon Submission

### Before Uploading to Git:

1. **Upload your model to Hugging Face**:
   ```bash
   python3 ../scripts/upload_model_to_huggingface.py \
     --model ../models/FINAL_BEST_MODEL_*.pth \
     --repo-id YOUR_USERNAME/chess-bot-model
   ```

2. **Hardcode repository in `src/main.py`**:
   - Open `Ding-Bot/src/main.py`
   - Find line: `HUGGINGFACE_MODEL_REPO = 'YOUR_USERNAME/chess-bot-model'`
   - Replace with: `HUGGINGFACE_MODEL_REPO = 'YOUR_USERNAME/chess-bot-model'` (your actual repo)

3. **Test locally**:
   ```bash
   python3 serve.py
   # Model should auto-download from Hugging Face
   ```

4. **Verify `.gitignore` excludes**:
   - `models/*.pth` (model files)

### What to Include in Git:

✅ **Include**:
- All source code (`src/`, `serve.py`)
- `requirements.txt`
- `README.md`
- `SUBMISSION.md` (this file)
- `.env.example` (template, not your actual `.env`)
- `start.sh` (startup script)
- `.gitignore` (excludes models and .env)

❌ **Exclude** (handled by `.gitignore`):
- `models/*.pth` (model files - too large)
- `.env` (your personal config)

### For Evaluators:

Evaluators should:
1. Clone your repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run `python3 serve.py`
4. Model will auto-download from the hardcoded Hugging Face repository on first run

## Troubleshooting

**Model not downloading?**
- Check `HUGGINGFACE_MODEL_REPO` is set correctly
- Verify repository is public (or provide access token)
- Check internet connection
- Check `huggingface_hub` is installed: `pip install huggingface_hub`

**Model not found locally?**
- This is expected! Model should download from Hugging Face automatically
- Check `models/` directory after first run

**Port already in use?**
- Change port: `SERVE_PORT=5059 python3 serve.py`
- Or kill existing process on port 5058

