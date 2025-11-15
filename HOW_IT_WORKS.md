# How Ding-Bot Finds Your Hugging Face Model

## Model Loading Flow

```
Ding-Bot Startup
    ↓
1. Check for local model in models/ directory
    ↓ (if not found)
2. Check HUGGINGFACE_MODEL_REPO environment variable
    ↓ (if set)
3. Download model from Hugging Face Hub
    ↓
4. Save to models/ directory (cached)
    ↓
5. Load model and start engine
```

## Environment Variable Setup

Ding-Bot reads `HUGGINGFACE_MODEL_REPO` from:

1. **`.env` file** (if exists):
   ```bash
   HUGGINGFACE_MODEL_REPO=your-username/chess-bot-model
   ```

2. **Environment variable**:
   ```bash
   export HUGGINGFACE_MODEL_REPO='your-username/chess-bot-model'
   ```

3. **Startup script** (`start.sh`):
   - Automatically loads `.env` file
   - Checks if variable is set
   - Shows helpful error if missing

## Code Location

The model loading logic is in `src/main.py`:

```python
# Line 95: Read environment variable
HUGGINGFACE_MODEL_REPO = os.environ.get('HUGGINGFACE_MODEL_REPO', None)

# Lines 182-201: Model loading logic
# 1. Check local models/
# 2. If not found, check HUGGINGFACE_MODEL_REPO
# 3. Download from Hugging Face if set
# 4. Load model

# Lines 98-147: download_model_from_huggingface() function
# Uses huggingface_hub library to download model
```

## For Hackathon Submission

**You need to:**
1. Upload model to Hugging Face (public repository)
2. Set `HUGGINGFACE_MODEL_REPO` in `.env` or environment
3. Include `.env.example` in Git (template for evaluators)

**Evaluators need to:**
1. Copy `.env.example` to `.env`
2. Set `HUGGINGFACE_MODEL_REPO` to your repository
3. Run `./start.sh` or `python3 serve.py`
4. Model auto-downloads on first run

## Example

```bash
# Your Hugging Face repository
HUGGINGFACE_MODEL_REPO=john-doe/chess-bot-model

# Ding-Bot will:
# 1. Check models/ (empty on first run)
# 2. Find HUGGINGFACE_MODEL_REPO='john-doe/chess-bot-model'
# 3. Download from https://huggingface.co/john-doe/chess-bot-model
# 4. Save to models/FINAL_BEST_MODEL_*.pth
# 5. Load and use model
```

## Troubleshooting

**"Model not found" error:**
- Check `HUGGINGFACE_MODEL_REPO` is set correctly
- Verify repository name matches exactly (case-sensitive)
- Ensure repository is public (or provide access token)

**"No .pth files found" error:**
- Verify model was uploaded to Hugging Face
- Check repository contains `.pth` file
- Try downloading manually: `python3 scripts/download_model_from_huggingface.py --repo-id YOUR_REPO`
