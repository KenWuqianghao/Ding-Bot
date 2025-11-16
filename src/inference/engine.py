"""Main chess engine interface."""
import sys
import os
from pathlib import Path

# CRITICAL: Add src/ to path BEFORE any other imports
# Handle both absolute and relative __file__ paths
if __file__:
    src_path = Path(__file__).parent.parent.resolve()
else:
    # Fallback: try to find src/ from current working directory
    src_path = Path.cwd() / 'src'
    if not src_path.exists():
        src_path = Path.cwd().parent / 'src'

# Add to path if not already there
src_path_str = str(src_path.resolve())
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

import torch
import chess
from typing import Optional
import os

from model.architecture import ChessNet, ChessNetLarge, ChessNetXL
from model.architectures.leela_style import LeelaChessNet, LeelaChessNetLarge
try:
    from model.architectures.leela_tiny import LeelaChessNetTiny
except ImportError:
    LeelaChessNetTiny = None
from engine.evaluation import NNEvaluator
from engine.stockfish_evaluator import StockfishEvaluator
from engine.ensemble_evaluator import EnsembleEvaluator
from engine.search import MinimaxSearch
from utils.board_utils import fen_to_board, is_game_over
from utils.time_management import allocate_time


def detect_model_size(checkpoint: dict) -> str:
    """
    Detect model size from checkpoint by examining architecture.
    
    Args:
        checkpoint: Model checkpoint dictionary
        
    Returns:
        'base', 'large', 'xl', 'leela-tiny', 'leela', or 'leela-large'
    """
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # PRIORITY 1: Check for Leela architecture FIRST (has conv_input, not conv1)
    # This must come before conv1 check to correctly identify Leela models
    if 'conv_input.weight' in state_dict:
        conv_input_out = state_dict['conv_input.weight'].shape[0]
        if conv_input_out == 96:
            return 'leela-tiny'  # LeelaChessNetTiny
        elif conv_input_out == 256:
            # Could be leela or leela-large - check number of residual blocks
            max_block = -1
            for k in state_dict.keys():
                if 'residual_blocks.' in k:
                    parts = k.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        max_block = max(max_block, int(parts[1]))
            num_blocks = max_block + 1 if max_block >= 0 else 6
            if num_blocks >= 10:
                return 'leela-large'
            else:
                return 'leela'
        elif conv_input_out == 384:
            return 'leela-large'
    
    # PRIORITY 2: Check for LeelaChessNetTiny by residual_blocks structure
    # (has residual_blocks.0, etc. with 96 channels)
    if 'residual_blocks.0.conv1.weight' in state_dict:
        residual_channels = state_dict['residual_blocks.0.conv1.weight'].shape[0]
        if residual_channels == 96:
            return 'leela-tiny'
        elif residual_channels == 256:
            # Check number of blocks
            max_block = -1
            for k in state_dict.keys():
                if 'residual_blocks.' in k:
                    parts = k.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        max_block = max(max_block, int(parts[1]))
            num_blocks = max_block + 1 if max_block >= 0 else 6
            if num_blocks >= 10:
                return 'leela-large'
            else:
                return 'leela'
        elif residual_channels == 384:
            return 'leela-large'
    
    # PRIORITY 3: Check for ChessNet architecture (has conv1, not conv_input)
    if 'conv1.weight' in state_dict:
        conv1_out_channels = state_dict['conv1.weight'].shape[0]
        if conv1_out_channels == 64:
            return 'base'
        elif conv1_out_channels == 128:
            return 'large'
        elif conv1_out_channels == 256:
            return 'xl'
        elif conv1_out_channels == 384:
            return 'leela-large'  # Shouldn't happen, but handle it
    
    # Fallback: check for residual block naming pattern (old ChessNet variants)
    if 'res_block1_conv.weight' in state_dict:
        # Check if it's XL (has more residual blocks)
        if 'res_block4_conv.weight' in state_dict:
            return 'xl'
        else:
            return 'large'
    
    # Default to base if can't determine
    return 'base'


class ChessEngine:
    """
    Main chess engine interface.
    
    Loads trained model and provides interface for move generation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        search_depth: int = 3,
        time_per_move: Optional[float] = None,
        device: Optional[torch.device] = None,
        model_size: Optional[str] = None,
        ensemble_paths: Optional[list] = None,
        use_stockfish: bool = False,
        stockfish_path: Optional[str] = None,
        stockfish_depth: int = 15
    ):
        """
        Initialize chess engine.
        
        Args:
            model_path: Path to saved model checkpoint (or first model if ensemble)
            search_depth: Default search depth for minimax
            time_per_move: Default time per move in seconds
            device: Device to run inference on
            model_size: Model size ('base', 'large', 'xl'). Auto-detected if None
            ensemble_paths: List of model paths for ensemble (if None, use single model)
            use_stockfish: If True, use Stockfish evaluator instead of NN (for testing)
            stockfish_path: Path to Stockfish executable (auto-detected if None)
            stockfish_depth: Stockfish search depth for evaluation
        """
        self.model_path = model_path
        # Don't force minimum depth - use what's requested (allows faster play)
        self.search_depth = search_depth
        self.time_per_move = time_per_move
        self.use_stockfish = use_stockfish
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Use Stockfish evaluator for testing
        if use_stockfish:
            import chess.engine
            # Find Stockfish executable
            if stockfish_path is None:
                possible_paths = ["stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]
                stockfish_path = None
                for path in possible_paths:
                    try:
                        test_engine = chess.engine.SimpleEngine.popen_uci(path)
                        test_engine.quit()
                        stockfish_path = path
                        break
                    except:
                        continue
                
                if stockfish_path is None:
                    raise FileNotFoundError("Stockfish executable not found. Please install Stockfish or specify stockfish_path")
            
            print(f"Using Stockfish evaluator: {stockfish_path} (depth {stockfish_depth})")
            stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.evaluator = StockfishEvaluator(stockfish_engine, depth=stockfish_depth)
            self.model = None  # No model needed for Stockfish
        else:
            # Load model(s)
            if ensemble_paths:
                # Ensemble mode
                models = []
                for path in ensemble_paths:
                    if not os.path.exists(path):
                        print(f"Warning: Model not found at {path}, skipping")
                        continue
                    
                    checkpoint = torch.load(path, map_location=self.device)
                    detected_size = detect_model_size(checkpoint) if model_size is None else model_size
                    
                    # Create model
                    if detected_size == 'large':
                        model = ChessNetLarge()
                    elif detected_size == 'xl':
                        model = ChessNetXL()
                    elif detected_size == 'leela-tiny':
                        if LeelaChessNetTiny is None:
                            raise ImportError("LeelaChessNetTiny not available. Check leela_tiny.py exists.")
                        # Detect number of residual blocks and channels from checkpoint
                        state_dict = checkpoint['model_state_dict']
                        residual_blocks = [k for k in state_dict.keys() if 'residual_blocks.' in k]
                        max_block = -1
                        for k in residual_blocks:
                            parts = k.split('.')
                            if len(parts) >= 2 and parts[1].isdigit():
                                max_block = max(max_block, int(parts[1]))
                        num_blocks = max_block + 1 if max_block >= 0 else 3  # Default to 3 for tiny
                        # Detect channels from conv_input
                        channels = 96  # Default
                        if 'conv_input.weight' in state_dict:
                            channels = state_dict['conv_input.weight'].shape[0]
                        model = LeelaChessNetTiny(num_residual_blocks=num_blocks, channels=channels)
                        print(f"Detected LeelaChessNetTiny: {num_blocks} blocks, {channels} channels")
                    elif detected_size == 'leela':
                        model = LeelaChessNet(num_residual_blocks=6)
                    elif detected_size == 'leela-large':
                        # Detect number of residual blocks from checkpoint
                        state_dict = checkpoint['model_state_dict']
                        residual_blocks = [k for k in state_dict.keys() if 'residual_blocks' in k]
                        max_block = -1
                        for k in residual_blocks:
                            parts = k.split('.')
                            if len(parts) >= 2 and parts[1].isdigit():
                                max_block = max(max_block, int(parts[1]))
                        num_blocks = max_block + 1 if max_block >= 0 else 12  # Default to 12 if can't detect
                        model = LeelaChessNetLarge(num_residual_blocks=num_blocks)
                    else:
                        model = ChessNet()
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    model.eval()
                    models.append(model)
            
                if not models:
                    raise RuntimeError("No valid models found for ensemble")
                
                print(f"Loaded ensemble of {len(models)} models")
                self.model = models[0]  # Keep first for compatibility
                self.evaluator = EnsembleEvaluator(models, self.device)
            else:
                # Single model mode
                if model_path is None:
                    raise ValueError("model_path is required when not using Stockfish evaluator")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found at {model_path}")
            
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Detect model size if not specified
                if model_size is None:
                    model_size = detect_model_size(checkpoint)
                
                # Create appropriate model
                if model_size == 'large':
                    self.model = ChessNetLarge()
                elif model_size == 'xl':
                    self.model = ChessNetXL()
                elif model_size == 'leela-tiny':
                    if LeelaChessNetTiny is None:
                        raise ImportError("LeelaChessNetTiny not available. Check leela_tiny.py exists.")
                    # Detect number of residual blocks and channels from checkpoint
                    state_dict = checkpoint['model_state_dict']
                    residual_blocks = [k for k in state_dict.keys() if 'residual_blocks.' in k]
                    max_block = -1
                    for k in residual_blocks:
                        parts = k.split('.')
                        if len(parts) >= 2 and parts[1].isdigit():
                            max_block = max(max_block, int(parts[1]))
                    num_blocks = max_block + 1 if max_block >= 0 else 3  # Default to 3 for tiny
                    # Detect channels from conv_input
                    channels = 96  # Default
                    if 'conv_input.weight' in state_dict:
                        channels = state_dict['conv_input.weight'].shape[0]
                    self.model = LeelaChessNetTiny(num_residual_blocks=num_blocks, channels=channels)
                    print(f"Detected LeelaChessNetTiny: {num_blocks} residual blocks, {channels} channels")
                elif model_size == 'leela':
                    self.model = LeelaChessNet(num_residual_blocks=6)
                elif model_size == 'leela-large':
                    # Detect number of residual blocks from checkpoint
                    state_dict = checkpoint['model_state_dict']
                    residual_blocks = [k for k in state_dict.keys() if 'residual_blocks' in k]
                    max_block = -1
                    for k in residual_blocks:
                        parts = k.split('.')
                        if len(parts) >= 2 and parts[1].isdigit():
                            max_block = max(max_block, int(parts[1]))
                    num_blocks = max_block + 1 if max_block >= 0 else 12  # Default to 12 if can't detect
                    self.model = LeelaChessNetLarge(num_residual_blocks=num_blocks)
                    print(f"Detected {num_blocks} residual blocks in checkpoint")
                else:
                    self.model = ChessNet()
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Handle quantized models (FP16)
                quantization = checkpoint.get('quantization', None)
                if quantization == 'fp16':
                    self.model = self.model.half()  # Convert to FP16
                    print(f"Loaded FP16 quantized model")
                
                self.model.to(self.device)
                self.model.eval()
                
                # Initialize evaluator
                self.evaluator = NNEvaluator(self.model, self.device)
        
        # Use requested search depth (don't force minimum - allows faster play)
        self.searcher = MinimaxSearch(
            self.evaluator,
            max_depth=search_depth,
            time_limit=time_per_move
        )
    
    def get_best_move(
        self,
        fen: str,
        time_limit: Optional[float] = None
    ) -> Optional[str]:
        """
        Get best move for a position.
        
        Args:
            fen: FEN string of current position
            time_limit: Optional time limit in seconds
            
        Returns:
            UCI move string
        """
        board = fen_to_board(fen)
        
        # Check if game is over
        if is_game_over(board):
            return None
        
        # Check if model is available (critical dependency)
        if self.evaluator is None:
            raise RuntimeError("Evaluator not available - engine cannot function")
        
        if not self.use_stockfish and self.model is None:
            raise RuntimeError("Neural network not available - engine cannot function")
        
        # Use provided time limit or default
        search_time = time_limit if time_limit is not None else self.time_per_move
        
        # Run search
        if search_time:
            self.searcher.time_limit = search_time
            best_move = self.searcher.iterative_deepening(board, search_time)
        else:
            best_move, _ = self.searcher.search(
                board,
                self.searcher.max_depth,  # Use searcher's max_depth
                alpha=-float('inf'),
                beta=float('inf'),
                ply=0
            )
        
        # Validate move is legal
        if best_move and best_move in board.legal_moves:
            return best_move.uci()
        else:
            # Fallback to first legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0].uci()
            return None
    
    def validate_move(self, fen: str, move: str) -> bool:
        """
        Validate that a move is legal.
        
        Args:
            fen: FEN string of position
            move: UCI move string
            
        Returns:
            True if move is legal
        """
        board = fen_to_board(fen)
        try:
            chess_move = chess.Move.from_uci(move)
            return chess_move in board.legal_moves
        except ValueError:
            return False
    
    def uci_loop(self):
        """
        UCI protocol loop.
        
        Handles UCI commands:
        - uci: Identify as UCI engine
        - isready: Check if ready
        - position: Set position
        - go: Calculate best move
        - quit: Exit
        """
        board = chess.Board()
        
        while True:
            try:
                command = input().strip()
            except EOFError:
                break
            
            if command == "uci":
                print("id name ChessBot")
                print("id author ChessHacks")
                print("uciok")
            
            elif command == "isready":
                print("readyok")
            
            elif command.startswith("position"):
                parts = command.split()
                if len(parts) < 2:
                    continue
                
                if parts[1] == "startpos":
                    board = chess.Board()
                    if len(parts) > 2 and parts[2] == "moves":
                        for move_str in parts[3:]:
                            move = chess.Move.from_uci(move_str)
                            if move in board.legal_moves:
                                board.push(move)
                elif parts[1] == "fen":
                    fen_str = " ".join(parts[2:8])
                    board = chess.Board(fen_str)
                    if len(parts) > 8 and parts[8] == "moves":
                        for move_str in parts[9:]:
                            move = chess.Move.from_uci(move_str)
                            if move in board.legal_moves:
                                board.push(move)
            
            elif command.startswith("go"):
                parts = command.split()
                time_limit = None
                
                # Parse time controls
                for i, part in enumerate(parts):
                    if part == "movetime" and i + 1 < len(parts):
                        time_limit = float(parts[i + 1]) / 1000.0  # Convert ms to seconds
                    elif part == "wtime" and i + 1 < len(parts) and board.turn == chess.WHITE:
                        total_time = float(parts[i + 1]) / 1000.0
                        moves_remaining = 20  # Estimate
                        time_limit = allocate_time(total_time, moves_remaining)
                    elif part == "btime" and i + 1 < len(parts) and board.turn == chess.BLACK:
                        total_time = float(parts[i + 1]) / 1000.0
                        moves_remaining = 20
                        time_limit = allocate_time(total_time, moves_remaining)
                
                # Get best move
                best_move = self.get_best_move(board.fen(), time_limit)
                if best_move:
                    print(f"bestmove {best_move}")
                else:
                    print("bestmove 0000")  # Null move
            
            elif command == "quit":
                break


def main():
    """Main entry point for UCI protocol."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chess Engine UCI Interface')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--search-depth',
        type=int,
        default=3,
        help='Default search depth'
    )
    
    args = parser.parse_args()
    
    engine = ChessEngine(
        args.model_path,
        search_depth=args.search_depth
    )
    
    engine.uci_loop()


if __name__ == '__main__':
    main()

