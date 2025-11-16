import sys
import os
from pathlib import Path

# Ding-Bot is now standalone - all code is in Ding-Bot/src
# Add Ding-Bot/src to path for imports - MUST BE FIRST
ding_bot_src_path = Path(__file__).parent.resolve()
ding_bot_src_path_str = str(ding_bot_src_path)

# CRITICAL: Remove Chess-Bot/src from path to avoid import conflicts
# Chess-Bot/src uses relative imports that break when imported as top-level
chess_bot_path = Path(__file__).parent.parent.parent.resolve()
chess_bot_src_path = str(chess_bot_path / 'src')

# Remove Chess-Bot paths to prevent conflicts
sys.path = [p for p in sys.path if p != chess_bot_src_path and p != str(chess_bot_path)]

# Insert Ding-Bot/src at the VERY BEGINNING (highest priority)
if ding_bot_src_path_str not in sys.path:
    sys.path.insert(0, ding_bot_src_path_str)
else:
    # Move to front if already in path
    sys.path.remove(ding_bot_src_path_str)
    sys.path.insert(0, ding_bot_src_path_str)

# Import utils - handle both relative and absolute imports
try:
    from .utils import chess_manager, GameContext
except ImportError:
    # Fallback for when running as script or when relative imports fail
    # Import directly from decorator module to avoid relative import issues
    utils_path = Path(__file__).parent / 'utils'
    if str(utils_path.parent) not in sys.path:
        sys.path.insert(0, str(utils_path.parent))
    # Import directly from decorator to avoid utils/__init__.py relative import issues
    from utils.decorator import chess_manager, GameContext

from chess import Move

# Import torch (check if available)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

# Import our chess engine (only if torch is available)
if TORCH_AVAILABLE:
    try:
        # CRITICAL: Ensure ding_bot_src_path is FIRST in sys.path
        # This must happen before any imports
        if ding_bot_src_path_str not in sys.path:
            sys.path.insert(0, ding_bot_src_path_str)
        
        # Verify data module exists
        data_path = Path(ding_bot_src_path) / 'data' / 'preprocessing.py'
        if not data_path.exists():
            print(f"Warning: data/preprocessing.py not found at {data_path}")
        
        # Now import should work - use direct import from Ding-Bot/src
        # The path setup in evaluation.py and engine.py will also run, but
        # having it here first ensures it works
        from inference.engine import ChessEngine
        ENGINE_AVAILABLE = True
            
    except Exception as import_err:
        print(f"Warning: Could not import ChessEngine: {import_err}")
        print(f"Ding-Bot src path: {ding_bot_src_path}")
        print(f"Ding-Bot src exists: {ding_bot_src_path.exists()}")
        print(f"Data dir exists: {(ding_bot_src_path / 'data').exists()}")
        print(f"Preprocessing exists: {(ding_bot_src_path / 'data' / 'preprocessing.py').exists()}")
        print(f"Current sys.path (first 5): {sys.path[:5]}")
        import traceback
        traceback.print_exc()
        ENGINE_AVAILABLE = False
        ChessEngine = None
else:
    ENGINE_AVAILABLE = False
    ChessEngine = None

# Global engine instance (loaded once)
engine = None
# Store engine loading error for debugging
engine_load_error = None

# Write code here that runs once
# Hugging Face model repository (hardcoded for submission)
# TODO: Replace with your actual Hugging Face repository ID
HUGGINGFACE_MODEL_REPO = 'KenWu/chess-bot-model'  # CHANGE THIS TO YOUR REPO
# Fallback to environment variable only if still using placeholder (for testing)
if HUGGINGFACE_MODEL_REPO == 'YOUR_USERNAME/chess-bot-model':
    HUGGINGFACE_MODEL_REPO = os.environ.get('HUGGINGFACE_MODEL_REPO', None)

def download_model_from_huggingface(repo_id: str, output_dir: str = "models", branch_priority: list = None) -> str:
    """
    Download model from Hugging Face Hub if not found locally.
    
    Args:
        repo_id: Hugging Face repository ID
        output_dir: Directory to save model
        branch_priority: List of model name patterns to prioritize (e.g., ['tiny_model', 'DISTILLED'])
    
    Returns:
        Path to downloaded model file, or None if download failed
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("WARNING: huggingface_hub not installed. Cannot download from Hugging Face.")
        print("Install with: pip install huggingface_hub")
        return None
    
    print(f"Downloading model from Hugging Face: {repo_id}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # List files in repo
        files = list_repo_files(repo_id=repo_id, repo_type="model")
        model_files = [f for f in files if f.endswith('.pth')]
        
        if not model_files:
            print(f"ERROR: No .pth files found in {repo_id}")
            return None
        
        # If branch_priority is provided, try to find matching model
        target_file = None
        if branch_priority:
            for priority_pattern in branch_priority:
                matching_files = [f for f in model_files if priority_pattern in f]
                if matching_files:
                    # For BASE_ADVANCED, prioritize opening-trained versions
                    if 'BASE_ADVANCED' in priority_pattern:
                        opening_files = [f for f in matching_files if '_openings_' in f]
                        if opening_files:
                            matching_files = opening_files
                    # Sort by name (most recent timestamp usually comes last alphabetically)
                    target_file = sorted(matching_files)[-1]
                    print(f"  Found branch-specific model matching '{priority_pattern}': {target_file}")
                    break
        
        # If no branch-specific model found, download latest model file (alphabetically last)
        if not target_file:
            target_file = sorted(model_files)[-1]
            print(f"  No branch-specific model found, downloading latest: {target_file}")
        
        print(f"  Downloading {target_file}...")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=target_file,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Model downloaded to: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"ERROR: Failed to download from Hugging Face: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load the chess model
def initialize_engine():
    """Initialize the chess engine with our trained model."""
    global engine
    
    # HARDCODED BRANCH: distilled-model
    # This file is for the distilled-model branch - prioritize distilled model models
    BRANCH_NAME = 'distilled-model'
    hf_download_priority = ['tiny_model', 'DISTILLED']  # Will match both openings and regular distilled model
    print(f"[BRANCH: {BRANCH_NAME}] Will prioritize Hugging Face models: {hf_download_priority}")
    
    # Priority: 1) Latest BASE_ADVANCED (current training), 2) Latest resumed model, 3) Latest epoch checkpoint, 4) Most recent .pth, 5) Download from Hugging Face
    # Check Ding-Bot/models first (for standalone deployment), then fall back to Chess-Bot/models (for local development)
    ding_bot_model_dir = os.path.join(Path(__file__).parent.parent, 'models')  # Ding-Bot/models
    chess_bot_model_dir = os.path.join(chess_bot_path, 'models') if os.path.exists(chess_bot_path) else None  # Chess-Bot/models (may not exist in judge system)
    
    # Use Ding-Bot/models if it exists (even if empty - for standalone deployment)
    # Otherwise use Chess-Bot/models (for development)
    if os.path.exists(ding_bot_model_dir):
        model_dir = ding_bot_model_dir
        print(f"Using Ding-Bot models directory: {model_dir}")
    else:
        model_dir = chess_bot_model_dir
        print(f"Using Chess-Bot models directory: {model_dir}")
    
    model_path = None
    
    if os.path.exists(model_dir):
        # HARDCODED BRANCH: distilled-model
        # Priority 1: Look for tiny_model or DISTILLED models
        if not model_path:
            tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
            if tiny_models:
                tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                model_path = os.path.join(model_dir, tiny_models[0])
                print(f"✓ [BRANCH: distilled-model] Using distilled model: {tiny_models[0]}")
                print(f"  This model was fine-tuned on opening positions for better opening play!")
        # Priority 2: Look for latest tiny_model or DISTILLED model
        if not model_path:
            tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
            if tiny_models:
                tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                model_path = os.path.join(model_dir, tiny_models[0])
                print(f"✓ [BRANCH: distilled-model] Using latest distilled model: {tiny_models[0]}")
        # Priority 3: Fallback to most recent .pth file (only if no distilled model found)
        if not model_path:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if model_files:
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                model_path = os.path.join(model_dir, model_files[0])
                print(f"Using most recent model: {model_files[0]}")
    
    # If no local model found, try downloading from Hugging Face
    # IMPORTANT: If using Ding-Bot/models (standalone), always try Hugging Face first
    # Only fall back to Chess-Bot/models if Hugging Face fails and we're in development mode
    if not model_path or not os.path.exists(model_path):
        # If we're using Ding-Bot/models (standalone deployment), try Hugging Face
        if model_dir == ding_bot_model_dir and HUGGINGFACE_MODEL_REPO:
            print(f"\nNo local model found in Ding-Bot/models. Attempting to download from Hugging Face...")
            print(f"  Repository: {HUGGINGFACE_MODEL_REPO}")
            print(f"  Repository URL: https://huggingface.co/{HUGGINGFACE_MODEL_REPO}")
            try:
                downloaded_path = download_model_from_huggingface(HUGGINGFACE_MODEL_REPO, model_dir, branch_priority=hf_download_priority)
                if downloaded_path and os.path.exists(downloaded_path):
                    # HARDCODED BRANCH: distilled-model
                    # After downloading, re-check for distilled model models first
                    tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
                    if tiny_models:
                        tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                        model_path = os.path.join(model_dir, tiny_models[0])
                        print(f"✓ [BRANCH: distilled-model] Found distilled model after download: {tiny_models[0]}")
                    else:
                        tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
                        if tiny_models:
                            tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                            model_path = os.path.join(model_dir, tiny_models[0])
                            print(f"✓ [BRANCH: distilled-model] Found distilled model after download: {tiny_models[0]}")
                    
                    if not model_path:
                        # Use downloaded model only if no distilled model found
                        model_path = downloaded_path
                        print(f"✓ Using downloaded model: {os.path.basename(downloaded_path)}")
                else:
                    # If download fails and Chess-Bot/models exists, try that as fallback
                    if os.path.exists(chess_bot_model_dir):
                        print(f"\nHugging Face download failed. Falling back to Chess-Bot/models...")
                        model_dir = chess_bot_model_dir
                        # HARDCODED BRANCH: distilled-model - prioritize distilled model
                        if os.path.exists(model_dir):
                            tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
                            if tiny_models:
                                tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                                model_path = os.path.join(model_dir, tiny_models[0])
                                print(f"✓ [BRANCH: distilled-model] Using fallback distilled model: {tiny_models[0]}")
                            else:
                                tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
                                if tiny_models:
                                    tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                                    model_path = os.path.join(model_dir, tiny_models[0])
                                    print(f"✓ [BRANCH: distilled-model] Using fallback distilled model: {tiny_models[0]}")
                            if not model_path:
                                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                                if model_files:
                                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                                    model_path = os.path.join(model_dir, model_files[0])
                                    print(f"Using fallback model: {model_files[0]}")
                    
                    if not model_path or not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"Model download from Hugging Face failed and no fallback model found. "
                            f"Repository: {HUGGINGFACE_MODEL_REPO}. "
                            f"Please verify the repository exists and contains a .pth file."
                        )
            except Exception as e:
                error_msg = str(e)
                print(f"ERROR during Hugging Face download: {error_msg}")
                # Try fallback to Chess-Bot/models if available
                if os.path.exists(chess_bot_model_dir):
                    print(f"\nTrying fallback to Chess-Bot/models...")
                    model_dir = chess_bot_model_dir
                    # HARDCODED BRANCH: distilled-model - prioritize distilled model
                    tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
                    if tiny_models:
                        tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                        model_path = os.path.join(model_dir, tiny_models[0])
                        print(f"✓ [BRANCH: distilled-model] Using fallback distilled model: {tiny_models[0]}")
                    else:
                        tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
                        if tiny_models:
                            tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                            model_path = os.path.join(model_dir, tiny_models[0])
                            print(f"✓ [BRANCH: distilled-model] Using fallback distilled model: {tiny_models[0]}")
                    if not model_path:
                        raise FileNotFoundError(
                            f"Failed to download model from Hugging Face and no fallback available. "
                            f"Repository: {HUGGINGFACE_MODEL_REPO}. "
                            f"Error: {error_msg}. "
                            f"Please verify: 1) Repository exists at https://huggingface.co/{HUGGINGFACE_MODEL_REPO}, "
                            f"2) Repository is public, 3) Repository contains .pth files, "
                            f"4) huggingface_hub is installed (pip install huggingface_hub)"
                        ) from e
                else:
                    raise FileNotFoundError(
                        f"Failed to download model from Hugging Face. "
                        f"Repository: {HUGGINGFACE_MODEL_REPO}. "
                        f"Error: {error_msg}. "
                        f"Please verify: 1) Repository exists at https://huggingface.co/{HUGGINGFACE_MODEL_REPO}, "
                        f"2) Repository is public, 3) Repository contains .pth files, "
                        f"4) huggingface_hub is installed (pip install huggingface_hub)"
                    ) from e
        elif HUGGINGFACE_MODEL_REPO:
            # Using Chess-Bot/models but still try Hugging Face if no model found
            print(f"\nNo local model found. Attempting to download from Hugging Face...")
            print(f"  Repository: {HUGGINGFACE_MODEL_REPO}")
            print(f"  Repository URL: https://huggingface.co/{HUGGINGFACE_MODEL_REPO}")
            try:
                downloaded_path = download_model_from_huggingface(HUGGINGFACE_MODEL_REPO, model_dir, branch_priority=hf_download_priority)
                if downloaded_path and os.path.exists(downloaded_path):
                    # HARDCODED BRANCH: distilled-model
                    # After downloading, re-check for distilled model models first
                    tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
                    if tiny_models:
                        tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                        model_path = os.path.join(model_dir, tiny_models[0])
                        print(f"✓ [BRANCH: distilled-model] Found distilled model after download: {tiny_models[0]}")
                    else:
                        tiny_models = [f for f in os.listdir(model_dir) if (f.startswith('tiny_model_') or f.startswith('DISTILLED_')) and f.endswith('.pth')]
                        if tiny_models:
                            tiny_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                            model_path = os.path.join(model_dir, tiny_models[0])
                            print(f"✓ [BRANCH: distilled-model] Found distilled model after download: {tiny_models[0]}")
                    
                    if not model_path:
                        # Use downloaded model only if no distilled model found
                        model_path = downloaded_path
                        print(f"✓ Using downloaded model: {os.path.basename(downloaded_path)}")
                else:
                    raise FileNotFoundError(
                        f"Model download from Hugging Face returned None or file doesn't exist. "
                        f"Repository: {HUGGINGFACE_MODEL_REPO}. "
                        f"Please verify the repository exists and contains a .pth file."
                    )
            except Exception as e:
                error_msg = str(e)
                print(f"ERROR during Hugging Face download: {error_msg}")
                raise FileNotFoundError(
                    f"Failed to download model from Hugging Face. "
                    f"Repository: {HUGGINGFACE_MODEL_REPO}. "
                    f"Error: {error_msg}. "
                    f"Please verify: 1) Repository exists at https://huggingface.co/{HUGGINGFACE_MODEL_REPO}, "
                    f"2) Repository is public, 3) Repository contains .pth files, "
                    f"4) huggingface_hub is installed (pip install huggingface_hub)"
                ) from e
        else:
            raise FileNotFoundError(
                f"Model not found in {model_dir} and HUGGINGFACE_MODEL_REPO is not set. "
                "Please train a model first, set HUGGINGFACE_MODEL_REPO in src/main.py, "
                "or provide model locally."
            )
    
    # Load neural network model
    print(f"Loading neural network model from {model_path}...")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required but not available")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    engine = ChessEngine(
        model_path=model_path,
        search_depth=4,  # Reduced from 6 to 4 for faster search (prevent timeouts) (adaptive depth handles time limits)
        # Adaptive time management ensures we stay within 1min/game:
        # - Opening (0.3s): caps to depth 2
        # - Middlegame (0.8-1.2s): can reach depth 3-4
        # - Endgame (1.5s): can reach depth 3-4
        time_per_move=1.0,  # Default, but will be overridden by time_limit
        device=device
    )
    
    print("Chess engine loaded successfully!")

# Initialize engine when module loads (only if available)
if ENGINE_AVAILABLE:
    try:
        initialize_engine()
        engine_load_error = None  # Clear any previous error
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        engine_load_error = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': error_details
        }
        print(f"Warning: Failed to load chess engine: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        print(error_details)
        print("Will use fallback random move selection")
        engine = None
else:
    engine_load_error = {
        'error': 'PyTorch not available',
        'type': 'ImportError',
        'traceback': 'Install dependencies: pip install torch'
    }
    print("Warning: Chess engine not available. Install dependencies: pip install torch")
    engine = None


@chess_manager.entrypoint
def make_move(ctx: GameContext):
    """
    Main entrypoint for making moves.
    Called every time the bot needs to make a move.
    """
    global engine, engine_load_error
    
    # Get legal moves (legal_moves is a property, not a method)
    legal_moves = list(ctx.board.legal_moves)
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # If engine failed to load, use random fallback
    if engine is None:
        print("=" * 80)
        print("WARNING: Engine not loaded, using random moves")
        print("=" * 80)
        if engine_load_error:
            print(f"Reason: {engine_load_error.get('error', 'Unknown error')}")
            print(f"Error type: {engine_load_error.get('type', 'Unknown')}")
            print("\nFull error details:")
            print(engine_load_error.get('traceback', 'No traceback available'))
        else:
            print("Reason: Engine initialization was skipped (ENGINE_AVAILABLE=False)")
        print("=" * 80)
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(move_probs)
        import random
        return random.choice(legal_moves)
    
    try:
        # Convert board to FEN
        fen = ctx.board.fen()
        
        # Calculate time limit from timeLeft (convert milliseconds to seconds)
        # AGGRESSIVE time management for 1-minute total time control
        time_left_seconds = ctx.timeLeft / 1000.0 if ctx.timeLeft > 0 else 1.0
        
        # Import time management utilities
        from src.utils.time_management import allocate_time_aggressive, is_critical_position, estimate_move_number
        
        # Determine if position is critical
        is_critical = is_critical_position(ctx.board)
        move_number = estimate_move_number(ctx.board)
        
        # Allocate time aggressively
        time_limit = allocate_time_aggressive(
            total_time_seconds=time_left_seconds,
            move_number=move_number,
            is_critical=is_critical,
            increment=0.0  # No increment in this format
        )
        
        # Debug output
        print(f"Time management: {time_left_seconds:.1f}s remaining, move {move_number}, "
              f"critical={is_critical}, allocated={time_limit:.2f}s")
        
        # Get best move from engine
        best_move_uci = engine.get_best_move(fen, time_limit=time_limit)
        
        if not best_move_uci:
            # Fallback to first legal move
            best_move = legal_moves[0]
        else:
            best_move = Move.from_uci(best_move_uci)
            if best_move not in legal_moves:
                # Fallback if move is somehow invalid
                best_move = legal_moves[0]
        
        # Get move probabilities from search tree
        # We'll evaluate top moves to create a probability distribution
        move_probs = get_move_probabilities(ctx.board, engine, legal_moves)
        
        # Log probabilities
        ctx.logProbabilities(move_probs)
        
        print(f"Selected move: {best_move.uci()} (from {len(legal_moves)} legal moves)")
        
        return best_move
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("=" * 80)
        print(f"ERROR in make_move: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\nFull traceback:")
        print(error_details)
        print("=" * 80)
        
        # Fallback to random move
        print("Falling back to random move selection")
        import random
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(move_probs)
        return random.choice(legal_moves)


def get_move_probabilities(board, engine, legal_moves):
    """
    Get probability distribution over legal moves.
    Uses model evaluation to weight moves.
    """
    move_scores = {}
    
    try:
        # Evaluate each move quickly
        for move in legal_moves:
            # Make move
            board.push(move)
            fen = board.fen()
            
            # Get evaluation
            try:
                eval_score = engine.evaluator.evaluate(board)
                # Convert to positive score for probability weighting
                # Higher eval = higher probability
                move_scores[move] = eval_score + 1000  # Shift to positive
            except:
                move_scores[move] = 500  # Default score
            
            # Undo move
            board.pop()
        
        # Convert scores to probabilities using softmax-like distribution
        if move_scores:
            # Normalize to probabilities
            min_score = min(move_scores.values())
            max_score = max(move_scores.values())
            
            if max_score == min_score:
                # All moves equal - uniform distribution
                return {move: 1.0 / len(legal_moves) for move in legal_moves}
            
            # Scale and normalize
            total = sum(move_scores.values())
            move_probs = {
                move: score / total
                for move, score in move_scores.items()
            }
            
            return move_probs
        else:
            # Fallback to uniform
            return {move: 1.0 / len(legal_moves) for move in legal_moves}
            
    except Exception as e:
        print(f"Error calculating move probabilities: {e}")
        # Fallback to uniform distribution
        return {move: 1.0 / len(legal_moves) for move in legal_moves}


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins.
    Clear any caches or reset model state.
    """
    global engine
    
    print("New game started - resetting...")
    
    # Clear search transposition table if engine is loaded
    if engine is not None and hasattr(engine, 'searcher'):
        if hasattr(engine.searcher, 'transposition_table'):
            engine.searcher.transposition_table.clear()
        if hasattr(engine.searcher, 'killer_moves'):
            engine.searcher.killer_moves.clear()
        if hasattr(engine.searcher, 'history'):
            engine.searcher.history.clear()
    
    print("Reset complete")
