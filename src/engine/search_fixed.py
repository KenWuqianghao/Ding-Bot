"""Fixed negamax search implementation."""

import chess
import time
import math
from typing import Tuple, Optional, Dict, List
from collections import defaultdict

from .evaluation import NNEvaluator
from .move_generation import get_legal_moves, order_moves_by_captures


def negamax_search(
    evaluator: NNEvaluator,
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    nodes_searched: List[int],
    ply: int = 0
) -> Tuple[Optional[chess.Move], float]:
    """
    Pure negamax search implementation.
    Always maximizes from current player's perspective.
    """
    nodes_searched[0] += 1
    
    # Terminal conditions
    if board.is_checkmate():
        return None, -10000 + ply  # Current player lost
    if board.is_stalemate() or board.is_insufficient_material():
        return None, 0.0
    
    # Leaf node: evaluate position
    if depth == 0:
        score = evaluator.evaluate(board)
        return None, score
    
    # Get legal moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        score = evaluator.evaluate(board)
        return None, score
    
    best_move = None
    best_score = -math.inf
    
    # Try each move
    for move in legal_moves:
        board.push(move)
        
        # Recursive search: negate and swap bounds
        _, score = negamax_search(
            evaluator,
            board,
            depth - 1,
            -beta,
            -alpha,
            nodes_searched,
            ply + 1
        )
        score = -score  # Negate from opponent's perspective
        
        board.pop()
        
        # Update best move
        if score > best_score:
            best_score = score
            best_move = move
        
        # Update alpha
        alpha = max(alpha, score)
        
        # Beta cutoff
        if score >= beta:
            break
    
    return best_move, best_score

