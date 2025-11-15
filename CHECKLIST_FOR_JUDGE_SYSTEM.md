# Checklist for Judge System - Model Loading

## Pre-Submission Checklist

### 1. Verify Hugging Face Repository

- [ ] Repository exists: https://huggingface.co/KenWu/chess-bot-model
- [ ] Repository is **PUBLIC** (not private)
- [ ] Repository contains at least one `.pth` file
- [ ] Model file is uploaded and visible in repository

**To verify:**
```bash
# Visit in browser
https://huggingface.co/KenWu/chess-bot-model

# Or use Python (if huggingface_hub installed)
python3 -c "from huggingface_hub import list_repo_files; print(list_repo_files('KenWu/chess-bot-model', repo_type='model'))"
```

### 2. Verify Code Configuration

- [ ] `src/main.py` line ~96: `HUGGINGFACE_MODEL_REPO = 'KenWu/chess-bot-model'`
- [ ] Repository format is correct: `username/repo-name` (no extra spaces, quotes, etc.)
- [ ] No placeholder values remain

### 3. Verify Dependencies

- [ ] `requirements.txt` includes `huggingface_hub>=0.20.0`
- [ ] All dependencies are installable: `pip install -r requirements.txt`

### 4. Test Download Process

**Simulate judge system (no local models):**
```bash
cd Ding-Bot
# Remove local models temporarily
mv models models_backup
mkdir models

# Test download
python3 test_huggingface_download.py

# Restore models
rm -rf models
mv models_backup models
```

### 5. Common Issues and Fixes

**Issue: "No .pth files found"**
- Fix: Upload model to Hugging Face repository
- Command: `python3 ../scripts/upload_model_to_huggingface.py --model ../models/FINAL_BEST_MODEL_*.pth --repo-id KenWu/chess-bot-model`

**Issue: "Repository not found"**
- Fix: Check repository name is correct (case-sensitive)
- Verify: https://huggingface.co/KenWu/chess-bot-model exists

**Issue: "Repository is private"**
- Fix: Make repository public in Hugging Face settings

**Issue: "huggingface_hub not installed"**
- Fix: Ensure `requirements.txt` includes `huggingface_hub>=0.20.0`
- Judge system should install from requirements.txt

**Issue: "Model file not found locally"**
- This is EXPECTED in judge system - model should download automatically
- If download fails, check above issues

## Model Loading Flow in Judge System

```
1. Judge system clones repository
2. Installs dependencies: pip install -r requirements.txt
3. Runs: python3 serve.py
4. Ding-Bot/src/main.py initializes:
   a. Checks for local model in models/ (empty in judge system)
   b. Reads HUGGINGFACE_MODEL_REPO from src/main.py
   c. Calls download_model_from_huggingface()
   d. Downloads model from Hugging Face
   e. Saves to models/ directory
   f. Loads model and starts engine
```

## Debugging Tips

If model loading fails in judge system:

1. **Check logs** for error messages
2. **Verify repository** is accessible: https://huggingface.co/KenWu/chess-bot-model
3. **Test locally** with no local models (simulate judge system)
4. **Check file format** - ensure .pth file is in repository
5. **Verify dependencies** - huggingface_hub must be installed

## Quick Test Command

```bash
# Test the exact flow judge system will use
cd Ding-Bot
rm -rf models && mkdir models
python3 serve.py
# Should see: "Downloading model from Hugging Face: KenWu/chess-bot-model"
# Should see: "✓ Model downloaded to: ..."
# Should see: "Chess engine loaded successfully!"
```

