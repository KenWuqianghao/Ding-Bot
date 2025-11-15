"""Legal move generation and ordering utilities."""
import chess
from typing import List, Dict


def get_legal_moves(board: chess.Board) -> List[chess.Move]:
    """
    Get all legal moves for a position.
    
    Args:
        board: chess.Board object
        
    Returns:
        Sorted list of legal moves
    """
    legal_moves = list(board.legal_moves)
    # Sort for consistency
    legal_moves.sort(key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
    return legal_moves


def order_moves_by_policy(
    moves: List[chess.Move],
    policy_probs: Dict[chess.Move, float]
) -> List[chess.Move]:
    """
    Sort moves by policy head probabilities (highest first).
    
    Args:
        moves: List of legal moves
        policy_probs: Dictionary mapping moves to probabilities
        
    Returns:
        Sorted list of moves (highest probability first)
    """
    # Sort by probability (descending)
    sorted_moves = sorted(
        moves,
        key=lambda m: policy_probs.get(m, 0.0),
        reverse=True
    )
    return sorted_moves


def order_moves_by_captures(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    """
    Order moves by capture value (captures first, then by piece value).
    
    Args:
        board: chess.Board object
        moves: List of legal moves
        
    Returns:
        Sorted list of moves
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    def move_priority(move: chess.Move) -> tuple:
        # Captures first
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            capture_value = piece_values.get(captured_piece.piece_type, 0)
            return (1, capture_value)  # Higher is better
        return (0, 0)
    
    return sorted(moves, key=move_priority, reverse=True)

