# Chess Model Integration

This bot uses the trained chess neural network from the parent Chess-Bot project.

## Setup

1. **Install dependencies** (if not already installed):
```bash
cd Ding-Bot
pip install -r requirements.txt
pip install torch  # Add PyTorch if not in requirements
```

2. **Ensure model exists**:
   - The bot looks for `models/best_model.pth` in the parent Chess-Bot directory
   - If not found, it will try to use the most recent `.pth` file in `models/`
   - Update the `model_path` in `main.py` if your model is elsewhere

## How It Works

1. **Initialization**: When the module loads, it initializes the ChessEngine with your trained model
2. **Move Selection**: 
   - Converts the current board position to FEN
   - Uses the engine's search algorithm to find the best move
   - Calculates move probabilities by evaluating all legal moves
   - Returns the best move and logs probabilities
3. **Time Management**: Uses available time (`timeLeft`) intelligently (10% of available time, max 5s)

## Features

- ✅ Uses trained neural network for evaluation
- ✅ Advanced search with alpha-beta pruning, LMR, aspiration windows
- ✅ Move probability logging for analysis
- ✅ Automatic fallback if engine fails
- ✅ Clears search cache between games

## Configuration

Edit `main.py` to adjust:
- `search_depth`: Default search depth (currently 6)
- `time_per_move`: Default time per move (currently 2.0s)
- `model_path`: Path to your trained model

## Testing

Run the devtools to test:
```bash
cd devtools
npm run dev
```

The bot will automatically use your trained model for move selection!

