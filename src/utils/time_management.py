"""Time management utilities for chess engine."""
from typing import Optional
import chess


def allocate_time_aggressive(
    total_time_seconds: float,
    move_number: int,
    is_critical: bool = False,
    increment: float = 0.0
) -> float:
    """
    Aggressive time management for very short time controls (1 minute total).
    
    Strategy:
    - Opening (moves 1-10): Use very little time (~0.5s per move)
    - Middlegame (moves 11-30): Moderate time (~1-2s per move)
    - Endgame (moves 31+): More time for precision (~2-3s per move)
    - Critical positions (checks, captures): Extra time
    
    Args:
        total_time_seconds: Total time remaining in seconds
        move_number: Current move number (1-indexed)
        is_critical: True if position is critical (check, capture, etc.)
        increment: Time increment per move in seconds
        
    Returns:
        Time to allocate for this move in seconds
    """
    # Estimate moves remaining (conservative)
    if move_number <= 10:
        moves_remaining = 40  # Opening: expect ~40 more moves
    elif move_number <= 30:
        moves_remaining = 30 - move_number  # Middlegame
    else:
        moves_remaining = max(10, 50 - move_number)  # Endgame
    
    # Base allocation strategy
    if move_number <= 10:
        # Opening: very fast, rely on opening knowledge
        base_time = 0.3
    elif move_number <= 20:
        # Early middlegame: moderate
        base_time = 0.8
    elif move_number <= 30:
        # Late middlegame: more time
        base_time = 1.2
    else:
        # Endgame: precision matters
        base_time = 1.5
    
    # Critical positions get extra time
    if is_critical:
        base_time *= 1.5
    
    # Proportional allocation: use more time if we have plenty
    # But be conservative - don't use more than 5% per move
    proportional = total_time_seconds / max(moves_remaining, 1)
    allocated = min(base_time, proportional * 0.8)  # Use 80% of proportional
    
    # Safety margin: always save 10% of total time
    max_allowed = total_time_seconds * 0.9
    allocated = min(allocated, max_allowed)
    
    # Minimum time (need at least 0.1s to think)
    allocated = max(allocated, 0.1)
    
    # Add increment
    allocated += increment
    
    return allocated


def is_critical_position(board: chess.Board) -> bool:
    """
    Check if position is critical and needs more thinking time.
    
    Args:
        board: chess.Board object
        
    Returns:
        True if position is critical
    """
    # Check if in check
    if board.is_check():
        return True
    
    # Check if there are captures available
    for move in board.legal_moves:
        if board.is_capture(move):
            return True
    
    # Check if there are checks available
    for move in board.legal_moves:
        if board.gives_check(move):
            return True
    
    return False


def estimate_move_number(board: chess.Board) -> int:
    """
    Estimate current move number from board state.
    
    Args:
        board: chess.Board object
        
    Returns:
        Estimated move number (1-indexed)
    """
    return board.fullmove_number


def allocate_time(total_time: float, moves_remaining: int, increment: float = 0.0) -> float:
    """
    Legacy time allocation function for compatibility.
    
    Args:
        total_time: Total time remaining in seconds
        moves_remaining: Estimated moves remaining
        increment: Time increment per move in seconds
        
    Returns:
        Time to allocate for this move in seconds
    """
    # Simple proportional allocation
    if moves_remaining <= 0:
        moves_remaining = 20  # Default estimate
    
    proportional = total_time / moves_remaining
    allocated = min(proportional * 0.8, total_time * 0.1)  # Use 80% of proportional, max 10% of total
    allocated = max(allocated, 0.1)  # Minimum 0.1s
    allocated += increment
    
    return allocated
