"""Stockfish-based evaluator for testing search framework."""

import chess
import chess.engine
from typing import Optional, Dict


class StockfishEvaluator:
    """
    Stockfish-based evaluator compatible with NNEvaluator interface.
    Used to test if search framework is working correctly.
    """
    """
    Stockfish-based evaluator that uses Stockfish's evaluation.
    Used to test if search framework is working correctly.
    """
    
    def __init__(self, stockfish_engine: chess.engine.SimpleEngine, depth: int = 15):
        """
        Initialize Stockfish evaluator.
        
        Args:
            stockfish_engine: Stockfish engine instance
            depth: Search depth for Stockfish evaluation
        """
        self.engine = stockfish_engine
        self.depth = depth
        # Cache evaluations to speed up repeated positions (e.g., in search)
        self.eval_cache: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using Stockfish.
        
        Args:
            board: chess.Board object
            
        Returns:
            Evaluation score in centipawns from current player's perspective
        """
        # Terminal conditions
        # board.is_checkmate() returns True if CURRENT player (board.turn) is checkmated
        # So if checkmated, current player lost - return very negative score
        if board.is_checkmate():
            return -10000  # Current player is checkmated (lost)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        # Check cache first (use FEN as key)
        fen = board.fen()
        if fen in self.eval_cache:
            self.cache_hits += 1
            cached_eval = self.eval_cache[fen]
            # Adjust for side to move (cache stores from white's perspective)
            if not board.turn:  # Black to move
                return -cached_eval
            return cached_eval
        
        self.cache_misses += 1
        
        # Get Stockfish evaluation
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].white()
        
        # Convert to centipawns
        if score.is_mate():
            mate_moves = score.mate()
            if mate_moves:
                # Prefer shorter mates
                mate_score = 10000 - abs(mate_moves) * 100
                return mate_score if mate_moves > 0 else -mate_score
            return 0
        
        cp = score.score()
        if cp is None:
            return 0
        
        # Stockfish returns score from white's perspective
        # Convert to centipawns (Stockfish uses pawns * 10)
        centipawns = cp / 10.0
        
        # Cache the evaluation (from white's perspective)
        # Limit cache size to avoid memory issues
        if len(self.eval_cache) < 10000:
            self.eval_cache[fen] = centipawns
        
        # For negamax search, we need evaluation from current player's perspective
        # If black to move, negate the evaluation
        if not board.turn:  # Black to move
            centipawns = -centipawns
        
        return centipawns
    
    def _material_eval(self, board: chess.Board) -> float:
        """Material evaluation (not used, but kept for interface compatibility)."""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    material += value
                else:
                    material -= value
        
        return material

