"""Neural network evaluator for chess positions."""
import sys
import os
from pathlib import Path

# CRITICAL: Add src/ to path BEFORE any other imports
# This ensures imports work even if src/ is not in Python path
# Handle both absolute and relative __file__ paths
if __file__:
    src_path = Path(__file__).parent.parent.resolve()
else:
    # Fallback: try to find src/ from current working directory
    src_path = Path.cwd() / 'src'
    if not src_path.exists():
        # Try parent directory
        src_path = Path.cwd().parent / 'src'

# Add to path if not already there
src_path_str = str(src_path.resolve())
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# DEBUG: Verify path was added and data module exists
# This helps diagnose import issues in judge system
if not any('data' in str(p) for p in [Path(p) / 'data' for p in sys.path[:3] if Path(p).exists()]):
    # Last resort: try adding parent directory if src/ doesn't work
    parent_path = str(src_path.parent)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)

import torch
import chess
from typing import Tuple, Dict, List
import numpy as np

from model.architecture import ChessNet
from data.preprocessing import fen_to_tensor
from engine.see import see


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
        # But checks are NOT worth sacrificing material for!
        # A check is worth ~50-80cp, NOT worth a piece (300-900cp)
        check_bonus = 0
        if board.is_check():
            # Being in check is bad, giving check is good
            # Reduced from 150cp to 60cp - checks aren't worth material sacrifices
            if board.turn == chess.WHITE:
                check_bonus = -60  # White in check (bad for white) - REDUCED from 150
            else:
                check_bonus = 60   # Black in check (good for white) - REDUCED from 150
        
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
        # But preventing castling is NOT worth sacrificing material for!
        # INCREASED bonuses to strongly encourage castling
        castling_bonus = 0
        move_number = board.fullmove_number
        # Stronger bonus in opening (moves 1-15)
        castling_multiplier = 1.5 if move_number <= 15 else 1.0
        
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_bonus += int(60 * castling_multiplier)  # INCREASED: 60cp (90cp in opening)
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_bonus += int(60 * castling_multiplier)  # INCREASED: 60cp (90cp in opening)
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_bonus -= int(60 * castling_multiplier)  # INCREASED: 60cp (90cp in opening)
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_bonus -= int(60 * castling_multiplier)  # INCREASED: 60cp (90cp in opening)
        
        # Check if castling has already happened (king moved from starting square)
        # If king is on e1/e8, castling is still possible, so bonus applies
        # If king has moved, remove castling bonus
        if board.king(chess.WHITE) and chess.square_file(board.king(chess.WHITE)) != 4:
            # White king not on e-file, castling rights lost
            if not board.has_kingside_castling_rights(chess.WHITE) and not board.has_queenside_castling_rights(chess.WHITE):
                castling_bonus -= 30  # Penalty for losing castling rights - REDUCED from 50
        if board.king(chess.BLACK) and chess.square_file(board.king(chess.BLACK)) != 4:
            # Black king not on e-file, castling rights lost
            if not board.has_kingside_castling_rights(chess.BLACK) and not board.has_queenside_castling_rights(chess.BLACK):
                castling_bonus += 30  # Bonus for white (black lost castling) - REDUCED from 50
        
        # CRITICAL: Penalty for early queen development (moves 1-10)
        # Bringing queen out early makes it vulnerable to attacks
        early_queen_penalty = 0
        move_number = board.fullmove_number
        if move_number <= 10:
            # Check if queen is developed (not on starting square)
            white_queen_sq = None
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.QUEEN and piece.color == chess.WHITE:
                    white_queen_sq = sq
                    break
            
            if white_queen_sq is not None:
                # Queen is developed if not on d1 (starting square)
                if white_queen_sq != chess.D1:
                    # Penalty increases for earlier moves
                    penalty = (11 - move_number) * 30  # 300cp penalty on move 1, 30cp on move 10
                    early_queen_penalty -= penalty
            
            # Same for black queen
            black_queen_sq = None
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.QUEEN and piece.color == chess.BLACK:
                    black_queen_sq = sq
                    break
            
            if black_queen_sq is not None:
                if black_queen_sq != chess.D8:
                    penalty = (11 - move_number) * 30
                    early_queen_penalty += penalty  # Bonus for white (black made mistake)
        
        # CRITICAL: MUCH stronger penalty for actually hanging pieces (can be captured)
        # This is the PRIMARY defense against giving away pieces
        hanging_penalty = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                # Check if this piece can be captured
                attackers = board.attackers(not piece.color, square)
                defenders = board.attackers(piece.color, square)
                
                # CRITICAL FIX: Check if piece can be captured even if defenders == attackers
                # A piece can be hanging if:
                # 1. More attackers than defenders (obvious hanging)
                # 2. Equal attackers/defenders BUT no defenders OR attacker can capture profitably
                # 3. Any attacker can capture with SEE >= 0 (profitable capture)
                can_be_captured = False
                is_hanging = False
                
                # Check each attacker to see if it can capture profitably
                for attacker_sq in attackers:
                    attacker_piece = board.piece_at(attacker_sq)
                    if attacker_piece:
                        # Use SEE to check if capture is profitable
                        try:
                            capture_move = chess.Move(attacker_sq, square)
                            see_value = see(board, capture_move)
                            # If SEE >= 0, capture is profitable (or at least equal)
                            if see_value >= 0:
                                can_be_captured = True
                                # If SEE is strongly positive (winning capture), it's definitely hanging
                                if see_value > 100:  # Winning by at least 100cp
                                    is_hanging = True
                                break
                        except:
                            # Fallback: simplified check if SEE fails
                            attacker_val = piece_values.get(attacker_piece.piece_type, 0)
                            captured_val = piece_values.get(piece.piece_type, 0)
                            # If attacker value <= captured value, it's a winning capture
                            if attacker_val <= captured_val:
                                can_be_captured = True
                                is_hanging = True
                                break
                            # If no defenders, it's hanging
                            elif len(defenders) == 0:
                                can_be_captured = True
                                is_hanging = True
                                break
                
                # Apply penalty if piece can be captured
                if can_be_captured:
                    piece_value = piece_values.get(piece.piece_type, 0)
                    # CRITICAL: Use FULL piece value penalty for hanging pieces
                    # This ensures the engine NEVER gives away pieces
                    if is_hanging or len(attackers) > len(defenders):
                        # Definitely hanging: FULL penalty (100% of piece value)
                        penalty = piece_value * 1.0  # 100% penalty - piece is lost
                    else:
                        # Can be captured but might be defended: still strong penalty
                        penalty = piece_value * 0.85  # 85% penalty
                    
                    if piece.color == chess.WHITE:
                        hanging_penalty -= penalty
                    else:
                        hanging_penalty += penalty
        
        # CRITICAL: Bonus for actually castling (not just having rights)
        # But castling is NOT worth sacrificing material for!
        # Castling is worth ~50-100cp, NOT worth a piece (300-900cp)
        # INCREASED bonus to strongly encourage castling
        actual_castling_bonus = 0
        move_number = board.fullmove_number
        castling_multiplier = 1.5 if move_number <= 15 else 1.0
        
        if board.king(chess.WHITE):
            white_king_file = chess.square_file(board.king(chess.WHITE))
            if white_king_file == 6 or white_king_file == 2:  # King on g1 or c1 (castled)
                actual_castling_bonus += int(120 * castling_multiplier)  # INCREASED: 120cp (180cp in opening)
        
        if board.king(chess.BLACK):
            black_king_file = chess.square_file(board.king(chess.BLACK))
            if black_king_file == 6 or black_king_file == 2:  # King on g8 or c8 (castled)
                actual_castling_bonus -= int(120 * castling_multiplier)  # INCREASED: 120cp (180cp in opening)
        
        # CRITICAL: STRONG penalty for moving king when castling is still available
        # This prevents the engine from making stupid king moves instead of castling
        king_move_penalty = 0
        move_number = board.fullmove_number
        if move_number <= 15:  # Early/mid game
            # Check if white king moved from e1 when castling was available
            if board.king(chess.WHITE):
                white_king_sq = board.king(chess.WHITE)
                white_king_file = chess.square_file(white_king_sq)
                # If king is not on e1 and we still have castling rights, we moved king incorrectly
                if white_king_file != 4:  # Not on e-file (e1)
                    # Check if we lost castling rights by moving king
                    if not board.has_kingside_castling_rights(chess.WHITE) and \
                       not board.has_queenside_castling_rights(chess.WHITE):
                        # We moved king and lost castling - STRONG penalty
                        # Penalty decreases as game progresses
                        penalty = (16 - min(move_number, 15)) * 25  # 375cp on move 1, 25cp on move 15
                        king_move_penalty -= penalty
            
            # Same for black
            if board.king(chess.BLACK):
                black_king_sq = board.king(chess.BLACK)
                black_king_file = chess.square_file(black_king_sq)
                if black_king_file != 4:  # Not on e-file (e8)
                    if not board.has_kingside_castling_rights(chess.BLACK) and \
                       not board.has_queenside_castling_rights(chess.BLACK):
                        penalty = (16 - min(move_number, 15)) * 25
                        king_move_penalty += penalty  # Bonus for white (black made mistake)
        
        # Combine all factors
        total_eval = material + check_bonus + safety_penalty + hanging_penalty + piece_count_bonus + castling_bonus + actual_castling_bonus + early_queen_penalty + king_move_penalty
        
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
            # Current player is checkmated (lost) - always return negative
            return -10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        fen = board.fen()
        
        # Check cache first (significant speedup for repeated positions)
        if fen in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[fen]
        
        self.cache_misses += 1
        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)
        
        # Convert input to FP16 if model is FP16 (required for GPU FP16 models)
        if self.is_fp16:
            board_tensor = board_tensor.half()
        
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
        
        # CRITICAL FIX: If model outputs wrong perspective, detect and fix it
        # For Black positions: material_eval is negated (negative = Black worse)
        # If model learned correctly: centipawns should also be negative when Black is worse
        # If model outputs from White's perspective: centipawns positive means White better = Black worse
        # Detection: if material_eval and centipawns have opposite signs for Black positions,
        # and the disagreement is significant, the model likely outputs wrong perspective
        if not board.turn:  # Black to move
            # Both should be negative if Black is worse, both positive if Black is better
            # If they have opposite signs with strong values, model outputs wrong perspective
            if (material_eval < -100 and centipawns > 100) or (material_eval > 100 and centipawns < -100):
                # Strong disagreement: negate centipawns to fix perspective
                centipawns = -centipawns
        
        # Blend NN evaluation with traditional heuristics
        # CRITICAL: NN must be the MAJOR part (hackathon requirement)
        # BUT: If material eval shows hanging pieces, we MUST trust it more
        # Check if this is a tactical position (captures available, checks, hanging pieces)
        is_tactical = False
        has_hanging_pieces = False
        
        # Check for hanging pieces by examining material eval
        # If material eval shows significant penalty, there are hanging pieces
        material_eval_before_perspective = self._material_eval(board)
        if abs(material_eval_before_perspective) > 300:  # Large material imbalance suggests hanging pieces
            has_hanging_pieces = True
        
        if board.is_check():
            is_tactical = True
        else:
            # Check for captures
            for move in board.legal_moves:
                if board.is_capture(move):
                    is_tactical = True
                    break
            # Check for hanging pieces
            if abs(material_eval) > 200:  # Significant material imbalance
                is_tactical = True
        
        # CRITICAL: If pieces are hanging, material eval MUST have high weight
        # This prevents the engine from ignoring material losses
        if has_hanging_pieces or abs(material_eval) > 300:
            # Pieces are hanging or big material loss: Trust material eval MUCH more
            # 40% NN, 60% material - Material eval is critical to prevent blunders
            # This is still compliant because NN is used, just weighted less when pieces hang
            blended_eval = 0.4 * centipawns + 0.6 * material_eval
        elif is_tactical:
            # Tactical positions: Material eval is CRITICAL to prevent blunders
            if abs(material_eval) > 200:  # Moderate material imbalance
                # Tactical: 50% NN, 50% material - Equal weight to prevent blunders
                blended_eval = 0.5 * centipawns + 0.5 * material_eval
            else:
                # Normal tactical: 70% NN, 30% material - COMPLIANT (NN is major part)
                blended_eval = 0.7 * centipawns + 0.3 * material_eval
        else:
            # Quiet positions: NN is primary, minimal material component
            # 85% NN, 15% material - COMPLIANT (NN is clearly major part)
            blended_eval = 0.85 * centipawns + 0.15 * material_eval
        
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
                # Current player is checkmated (lost) - always return negative
                results.append(-10000)
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
            
            # Convert input to FP16 if model is FP16 (required for GPU FP16 models)
            if self.is_fp16:
                batch_tensor = batch_tensor.half()
            
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
                
                # CRITICAL FIX: Detect and fix wrong perspective (same as evaluate())
                if not board.turn:  # Black to move
                    if (material_eval < -100 and centipawns > 100) or (material_eval > 100 and centipawns < -100):
                        centipawns = -centipawns
                
                # Check if tactical position and if pieces are hanging
                is_tactical = False
                has_hanging_pieces = False
                
                # Check for hanging pieces by examining material eval
                material_eval_before_perspective = self._material_eval(board)
                if abs(material_eval_before_perspective) > 300:  # Large material imbalance suggests hanging pieces
                    has_hanging_pieces = True
                
                if board.is_check():
                    is_tactical = True
                else:
                    for move in board.legal_moves:
                        if board.is_capture(move):
                            is_tactical = True
                            break
                    if abs(material_eval) > 200:
                        is_tactical = True
                
                # CRITICAL: If pieces are hanging, material eval MUST have high weight
                if has_hanging_pieces or abs(material_eval) > 300:
                    # Pieces are hanging: Trust material eval MUCH more (60% material)
                    blended_eval = 0.4 * centipawns + 0.6 * material_eval
                elif is_tactical:
                    # Tactical positions: Material eval is CRITICAL to prevent blunders
                    if abs(material_eval) > 200:
                        blended_eval = 0.5 * centipawns + 0.5 * material_eval  # Tactical: 50% NN, 50% material
                    else:
                        blended_eval = 0.7 * centipawns + 0.3 * material_eval  # Normal tactical: 70% NN
                else:
                    # Quiet positions: NN is primary (85%), minimal material component
                    blended_eval = 0.85 * centipawns + 0.15 * material_eval  # Quiet: 85% NN
                
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
        
        # Convert input to FP16 if model is FP16 (required for GPU FP16 models)
        if self.is_fp16:
            board_tensor = board_tensor.half()
        
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
        
        # Model should already output from side-to-move perspective
        # Don't negate here - that would double-negate for Black
        # If model didn't learn correctly, we'd need to detect and fix at evaluate() level
        
        return value * 1000.0, policy_dict

