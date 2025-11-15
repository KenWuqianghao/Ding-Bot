"""Neural network evaluator for chess positions."""
import sys
from pathlib import Path

# CRITICAL: Add src/ to path BEFORE any other imports
# This ensures imports work even if src/ is not in Python path
src_path = Path(__file__).parent.parent.resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import chess
from typing import Tuple, Dict, List
import numpy as np

from model.architecture import ChessNet
from data.preprocessing import fen_to_tensor


class NNEvaluator:
    """
    Neural network evaluator for chess positions.
    
    Converts board positions to tensors and runs model inference.
    """
    
    def __init__(self, model: ChessNet, device: torch.device = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained ChessNet model
            device: Device to run inference on (default: cuda if available, else cpu)
        """
        self.model = model
        self.model.eval()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        # Note: Traditional evaluator removed - NN must be primary (ChessHacks requirement)
        
        # Cache for evaluations (FEN -> score) to avoid redundant NN calls
        self.eval_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _material_eval(self, board: chess.Board) -> float:
        """
        Enhanced material evaluation with checks, captures, and piece safety bonuses.
        Returns evaluation from white's perspective in centipawns.
        """
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Basic material count
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    material += value
                else:
                    material -= value
        
        # CRITICAL: Add bonuses for checks (prevents ignoring checks)
        check_bonus = 0
        if board.is_check():
            # Being in check is bad, giving check is good
            if board.turn == chess.WHITE:
                check_bonus = -150  # White in check (bad for white) - INCREASED
            else:
                check_bonus = 150   # Black in check (good for white) - INCREASED
        
        # Add penalties for hanging pieces (pieces under attack without defense)
        safety_penalty = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                # Count attackers vs defenders
                attackers = board.attackers(not piece.color, square)
                defenders = board.attackers(piece.color, square)
                
                if len(attackers) > len(defenders):
                    # Piece is hanging (more attackers than defenders)
                    piece_value = piece_values.get(piece.piece_type, 0)
                    penalty = piece_value * 0.6  # 60% penalty for hanging pieces (INCREASED from 40%)
                    if piece.color == chess.WHITE:
                        safety_penalty -= penalty  # Penalty for white
                    else:
                        safety_penalty += penalty  # Bonus for white (black piece hanging)
        
        # Add bonus for having more pieces (activity/mobility)
        white_pieces = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
        black_pieces = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
        piece_count_bonus = (white_pieces - black_pieces) * 15
        
        # CRITICAL: Add castling bonuses (castling is good!)
        castling_bonus = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_bonus += 50  # White can castle kingside
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_bonus += 50  # White can castle queenside
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_bonus -= 50  # Black can castle kingside
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_bonus -= 50  # Black can castle queenside
        
        # Check if castling has already happened (king moved from starting square)
        # If king is on e1/e8, castling is still possible, so bonus applies
        # If king has moved, remove castling bonus
        if board.king(chess.WHITE) and chess.square_file(board.king(chess.WHITE)) != 4:
            # White king not on e-file, castling rights lost
            if not board.has_kingside_castling_rights(chess.WHITE) and not board.has_queenside_castling_rights(chess.WHITE):
                castling_bonus -= 50  # Penalty for losing castling rights
        if board.king(chess.BLACK) and chess.square_file(board.king(chess.BLACK)) != 4:
            # Black king not on e-file, castling rights lost
            if not board.has_kingside_castling_rights(chess.BLACK) and not board.has_queenside_castling_rights(chess.BLACK):
                castling_bonus += 50  # Bonus for white (black lost castling)
        
        # Combine all factors
        total_eval = material + check_bonus + safety_penalty + piece_count_bonus + castling_bonus
        
        return total_eval
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position using neural network with chess knowledge adjustment.
        
        Args:
            board: chess.Board object
            
        Returns:
            Evaluation score in centipawns
        """
        # Terminal conditions
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        fen = board.fen()
        
        # Check cache first (significant speedup for repeated positions)
        if fen in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[fen]
        
        self.cache_misses += 1
        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(board_tensor)
            nn_value = output['value'].item()
        
        # Denormalize: Convert from [-1, 1] back to centipawns
        # Models trained with tanh normalization: eval = tanh(eval / 600)
        #   -> Denorm: centipawns = 600 * arctanh(value)
        #   -> This allows models to learn extreme values effectively
        #
        # Old models used: eval = clamp(eval, -1000, 1000) / 1000
        #   -> Denorm: centipawns = value * 1000
        
        # Use arctanh denormalization for tanh-normalized models
        # Clamp value to avoid arctanh(±1) = inf
        nn_value_clamped = max(-0.9999, min(0.9999, nn_value))
        scale = 600.0  # Same scale used in training (dataset.py)
        try:
            centipawns = scale * np.arctanh(nn_value_clamped)
            # Check for NaN or inf
            if not np.isfinite(centipawns):
                # Fallback to linear scaling if arctanh fails
                centipawns = nn_value * 1000.0
        except (ValueError, OverflowError):
            # Fallback to linear scaling if arctanh fails
            centipawns = nn_value * 1000.0
        
        # TODO: Add checkpoint metadata to detect normalization method
        # For now, linear scaling works for both (new models will have slightly compressed range)
        
        # PERSPECTIVE HANDLING:
        # The dataset (dataset.py) adjusts evaluations for side-to-move during training:
        #   - If Black to move: negates eval (so model learns side-to-move perspective)
        #   - If White to move: keeps eval as-is
        # Therefore, the model SHOULD output from side-to-move perspective.
        # 
        # However, if the model didn't learn this correctly, it might output from White's perspective always.
        # We need to detect which case we're in.
        #
        # TEMPORARY FIX: Try WITHOUT negation first (assume model learned correctly)
        # If this causes issues, we can add back negation
        # 
        # if not board.turn:  # Black to move
        #     centipawns = -centipawns  # DISABLED - assume model outputs side-to-move perspective
        
        # Blend with material evaluation for stability
        # Note: We keep material eval as a small component to help undertrained models
        # But NN is still the primary evaluator (required by ChessHacks rules)
        material_eval = self._material_eval(board)
        
        # Material eval is from white's perspective
        # Model output (centipawns) should be from side-to-move perspective (if model learned correctly)
        # Adjust material eval to match model's perspective before blending
        if not board.turn:  # Black to move
            # If model outputs from side-to-move perspective, material eval should also be from Black's perspective
            material_eval = -material_eval  # Convert white's perspective to black's perspective
        
        # DRAMATICALLY increase traditional heuristic weight to prevent blunders
        # Model is still undertrained - prioritize safety over NN evaluation
        if abs(nn_value) < 0.1:
            # Undertrained: 30% NN, 70% traditional (heavily prioritize safety)
            blended_eval = 0.3 * centipawns + 0.7 * material_eval
        else:
            # Normal: 50% NN, 50% traditional (equal weight for safety)
            blended_eval = 0.5 * centipawns + 0.5 * material_eval
        
        # Cache the result (limit cache size to avoid memory issues)
        if len(self.eval_cache) < 10000:  # Limit cache to 10k entries
            self.eval_cache[fen] = blended_eval
        
        return blended_eval
    
    def evaluate_batch(self, boards: List[chess.Board]) -> List[float]:
        """
        Evaluate multiple positions in a single batch for efficiency.
        This is much faster than calling evaluate() multiple times.
        
        Args:
            boards: List of chess.Board objects to evaluate
            
        Returns:
            List of evaluation scores in centipawns
        """
        if not boards:
            return []
        
        # Check cache for each position
        results = []
        uncached_boards = []
        uncached_indices = []
        
        for idx, board in enumerate(boards):
            fen = board.fen()
            
            # Terminal conditions
            if board.is_checkmate():
                results.append(-10000 if board.turn == chess.WHITE else 10000)
                continue
            if board.is_stalemate() or board.is_insufficient_material():
                results.append(0.0)
                continue
            
            # Check cache
            if fen in self.eval_cache:
                self.cache_hits += 1
                results.append(self.eval_cache[fen])
            else:
                self.cache_misses += 1
                results.append(None)  # Placeholder
                uncached_boards.append(board)
                uncached_indices.append(idx)
        
        # Batch evaluate uncached positions
        if uncached_boards:
            # Convert all boards to tensors
            batch_tensors = []
            for board in uncached_boards:
                tensor = fen_to_tensor(board.fen()).unsqueeze(0)
                batch_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                output = self.model(batch_tensor)
                nn_values = output['value'].cpu().numpy()
            
            # Process each result
            for i, (board, nn_value) in enumerate(zip(uncached_boards, nn_values)):
                idx = uncached_indices[i]
                
                # Denormalize
                nn_value_clamped = max(-0.9999, min(0.9999, nn_value))
                scale = 600.0
                try:
                    centipawns = scale * np.arctanh(nn_value_clamped)
                    if not np.isfinite(centipawns):
                        centipawns = nn_value * 1000.0
                except (ValueError, OverflowError):
                    centipawns = nn_value * 1000.0
                
                # Blend with material eval
                material_eval = self._material_eval(board)
                if not board.turn:
                    material_eval = -material_eval
                
                # DRAMATICALLY increase traditional heuristic weight to prevent blunders
                # Model is still undertrained - prioritize safety over NN evaluation
                if abs(nn_value) < 0.1:
                    # Undertrained: 30% NN, 70% traditional (heavily prioritize safety)
                    blended_eval = 0.3 * centipawns + 0.7 * material_eval
                else:
                    # Normal: 50% NN, 50% traditional (equal weight for safety)
                    blended_eval = 0.5 * centipawns + 0.5 * material_eval
                
                # Cache result
                fen = board.fen()
                if len(self.eval_cache) < 10000:
                    self.eval_cache[fen] = blended_eval
                
                results[idx] = blended_eval
        
        return results
    
    def evaluate_with_policy(
        self,
        board: chess.Board
    ) -> Tuple[float, Dict[chess.Move, float]]:
        """
        Evaluate position and get policy distribution over moves.
        
        Args:
            board: chess.Board object
            
        Returns:
            Tuple of (value_score, policy_dict) where policy_dict maps
            moves to probabilities
        """
        fen = board.fen()
        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)
        legal_moves = list(board.legal_moves)
        
        with torch.no_grad():
            output = self.model(board_tensor)
            value = output['value'].item()
            policy_logits = output['policy'][0]  # [num_moves]
        
        # Convert logits to probabilities
        policy_probs = torch.softmax(policy_logits, dim=0)
        
        # Create policy dictionary
        # Note: This is simplified - full implementation would need
        # proper move encoding to map policy indices to moves
        policy_dict = {}
        
        # For now, distribute probability uniformly over legal moves
        # In a full implementation, you would:
        # 1. Encode each legal move to an index
        # 2. Look up probability from policy_probs
        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
        for move in legal_moves:
            policy_dict[move] = uniform_prob
        
        # Adjust value for side to move
        if not board.turn:
            value = -value
        
        return value * 1000.0, policy_dict

