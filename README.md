# Chess Bot - Neural Network Chess Engine

A neural network-based chess engine built for ChessHacks hackathon.

## Setup

### 1. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Node dependencies (for devtools)
cd devtools
npm install
```

### 2. Download Model

The trained model is automatically downloaded from Hugging Face if not found locally.

**The model repository is hardcoded in `src/main.py`** - no configuration needed!

**To set your repository:**
1. Open `src/main.py`
2. Find: `HUGGINGFACE_MODEL_REPO = 'YOUR_USERNAME/chess-bot-model'`
3. Replace with your repository: `HUGGINGFACE_MODEL_REPO = 'your-username/chess-bot-model'`

**Run server:**
```bash
python3 serve.py
# Model will auto-download on first run
```

**Manual Download (optional):**
```bash
# Download from Hugging Face manually
python3 scripts/download_model_from_huggingface.py \
  --repo-id your-username/chess-bot-model \
  --output-dir models
```

### 3. Run Server

```bash
# Start backend server
python3 serve.py

# Start frontend devtools (in another terminal)
cd devtools
npm run dev
```

Access the UI at: http://localhost:3000

## Model Information

- **Architecture**: LeelaChessNetLarge (12 residual blocks)
- **Parameters**: ~50-80M
- **Size**: ~367 MB (FP32)
- **Training**: Knowledge distillation from chess engine evaluations
- **Dataset**: Lichess positions with chess engine evaluations

## Model Loading

The engine automatically loads the latest model from `models/` directory:
1. Looks for `FINAL_BEST_MODEL_*.pth` (latest training)
2. Falls back to `*_epoch*.pth` (epoch checkpoints)
3. Falls back to most recent `.pth` file

## Testing

Run the server to test with your neural network model:
```bash
python3 serve.py
```

## Time Management

The engine uses aggressive time management for 1-minute time controls:
- Opening (moves 1-10): ~0.3s per move
- Middlegame (moves 11-30): ~0.8-1.2s per move
- Endgame (moves 31+): ~1.5s per move
- Critical positions: +50% time

## Performance

- **Search depth**: 4 (configurable)
- **Nodes searched**: 25-1200 per move (varies by position)
- **Move time**: 0.3-1.8s per move
- **Expected ELO**: 1800-2200+ (depends on training)

## File Structure

```
Ding-Bot/
├── src/
│   └── main.py          # Chess engine integration
├── serve.py             # FastAPI server
├── devtools/            # Frontend (Next.js)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Model Hosting

Models are hosted on Hugging Face Hub for easy distribution:

1. **Upload your model:**
   ```bash
   python3 scripts/upload_model_to_huggingface.py \
     --model models/FINAL_BEST_MODEL_*.pth \
     --repo-id your-username/chess-bot-model
   ```

2. **Hardcode repository in `src/main.py`:**
   - Open `src/main.py`
   - Set: `HUGGINGFACE_MODEL_REPO = 'your-username/chess-bot-model'`

3. **Ding-Bot will auto-download** if model not found locally

## Notes

- Model files (`*.pth`) are excluded from Git (too large for GitHub)
- Models are hosted on Hugging Face Hub for automatic download
- For hackathon submission, set `HUGGINGFACE_MODEL_REPO` environment variable
- Alternative: Use Git LFS or tiny model (<10MB) if preferred
