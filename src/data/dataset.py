"""Chess dataset for training neural network."""
import torch
from torch.utils.data import Dataset
import chess
from typing import List, Dict, Optional, Callable
import numpy as np

from .preprocessing import fen_to_tensor, move_to_index, uci_to_move


class ChessDataset(Dataset):
    """
    Dataset for chess positions with evaluations and best moves.
    
    Attributes:
        positions: List of FEN strings
        evaluations: List of centipawn scores (float)
        best_moves: List of UCI move strings (optional)
        transform: Optional augmentation function
    """
    
    def __init__(
        self,
        positions: List[str],
        evaluations: List[float],
        best_moves: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        normalize_value: bool = True
    ):
        """
        Initialize chess dataset.
        
        Args:
            positions: List of FEN strings
            evaluations: List of centipawn scores
            best_moves: Optional list of UCI move strings
            transform: Optional augmentation function
            normalize_value: Whether to normalize evaluations to [-1, 1]
        """
        self.positions = positions
        self.evaluations = evaluations
        self.best_moves = best_moves
        self.transform = transform
        self.normalize_value = normalize_value
        
        # Normalize evaluations to [-1, 1] range using tanh scaling
        # This preserves full range while mapping to [-1, 1]
        if normalize_value and len(evaluations) > 0:
            # Use tanh scaling: tanh(eval / scale) maps to [-1, 1]
            # Scale of 600 means ±600 cp maps to ±0.76, ±1000 cp maps to ±0.96
            # This allows learning extreme values while keeping outputs bounded
            scale = 600.0  # Adjust this to control sensitivity
            self.evaluations = [np.tanh(e / scale) for e in evaluations]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single data sample.
        
        Returns:
            Dictionary with keys:
            - 'board': tensor of shape [18, 8, 8]
            - 'value': float evaluation (normalized)
            - 'policy': optional tensor of shape [num_legal_moves] (one-hot if best_move provided)
            - 'legal_moves': list of legal moves
        """
        fen = self.positions[idx]
        board = chess.Board(fen)
        
        # Convert FEN to tensor
        board_tensor = fen_to_tensor(fen)
        
        # Get evaluation
        # Dataset evaluations are from WHITE's perspective (Lichess format)
        # Model needs to learn from SIDE TO MOVE's perspective
        # So we need to adjust: if Black to move, negate the evaluation
        eval_from_white = float(self.evaluations[idx])
        if board.turn == chess.BLACK:
            # Black to move: negate white's perspective to get black's perspective
            value = -eval_from_white
        else:
            # White to move: evaluation is already from white's perspective
            value = eval_from_white
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        
        # Create label dictionary
        label = {'value': value}
        
        # Create policy target if best move is provided
        policy = None
        if self.best_moves is not None and self.best_moves[idx]:
            policy = torch.zeros(len(legal_moves), dtype=torch.float32)
            best_move_uci = self.best_moves[idx]
            best_move = uci_to_move(best_move_uci, board)
            
            if best_move and best_move in legal_moves:
                move_idx = move_to_index(best_move, legal_moves)
                if move_idx >= 0:
                    policy[move_idx] = 1.0
            label['policy'] = policy
        
        # Apply augmentation if transform is provided
        if self.transform:
            board_tensor, label = self.transform(board_tensor, label)
            policy = label.get('policy', policy)
        
        result = {
            'board': board_tensor,
            'value': torch.tensor(label['value'], dtype=torch.float32),
        }
        
        # Policy: convert to 1968-dimensional vector (all possible moves)
        # For now, if policy exists, we'll create a uniform distribution
        # In a full implementation, we'd need move-to-index encoding
        if policy is not None:
            # Create 1968-dimensional policy (uniform over legal moves for now)
            policy_1968 = torch.zeros(1968, dtype=torch.float32)
            if len(legal_moves) > 0:
                uniform_prob = 1.0 / len(legal_moves)
                # For now, we can't map moves to indices, so skip policy training
                # Or use a simplified approach: just mark that policy exists
                result['has_policy'] = True
            else:
                result['has_policy'] = False
        else:
            result['has_policy'] = False
        
        # Don't include legal_moves in batch (it's variable length)
        # It's not needed for training anyway
        
        return result

