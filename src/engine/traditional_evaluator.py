"""Traditional chess evaluation function - can be blended with NN."""
import chess
from typing import Dict


class TraditionalEvaluator:
    """
    Strong traditional chess evaluation function.
    Can be blended with NN evaluation for hybrid approach.
    """
    
    # Piece values (centipawns)
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Piece-square tables (positional bonuses)
    # Values from white's perspective, flip for black
    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    KNIGHT_TABLE = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
    
    BISHOP_TABLE = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ]
    
    ROOK_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ]
    
    QUEEN_TABLE = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0,  0,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ]
    
    KING_MIDDLE_TABLE = [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
    
    KING_END_TABLE = [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
    
    def __init__(self):
        """Initialize traditional evaluator."""
        pass
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using traditional chess knowledge.
        
        Returns:
            Evaluation in centipawns from current player's perspective
        """
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        score = 0.0
        
        # Material
        score += self._material_eval(board)
        
        # Piece-square tables
        score += self._piece_square_eval(board)
        
        # Pawn structure
        score += self._pawn_structure_eval(board)
        
        # King safety
        score += self._king_safety_eval(board)
        
        # Mobility
        score += self._mobility_eval(board)
        
        # Adjust for side to move (negamax)
        if not board.turn:
            score = -score
        
        return score
    
    def _material_eval(self, board: chess.Board) -> float:
        """Material evaluation."""
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    material += value
                else:
                    material -= value
        return material
    
    def _piece_square_eval(self, board: chess.Board) -> float:
        """Piece-square table evaluation."""
        score = 0.0
        
        # Determine if endgame (for king table)
        material_count = sum(1 for sq in chess.SQUARES if board.piece_at(sq))
        is_endgame = material_count <= 12
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
            
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            # Flip for black pieces
            if piece.color == chess.BLACK:
                rank = 7 - rank
            
            table_idx = rank * 8 + file
            
            if piece.piece_type == chess.PAWN:
                bonus = self.PAWN_TABLE[table_idx]
            elif piece.piece_type == chess.KNIGHT:
                bonus = self.KNIGHT_TABLE[table_idx]
            elif piece.piece_type == chess.BISHOP:
                bonus = self.BISHOP_TABLE[table_idx]
            elif piece.piece_type == chess.ROOK:
                bonus = self.ROOK_TABLE[table_idx]
            elif piece.piece_type == chess.QUEEN:
                bonus = self.QUEEN_TABLE[table_idx]
            elif piece.piece_type == chess.KING:
                bonus = self.KING_END_TABLE[table_idx] if is_endgame else self.KING_MIDDLE_TABLE[table_idx]
            else:
                bonus = 0
            
            if piece.color == chess.WHITE:
                score += bonus
            else:
                score -= bonus
        
        return score
    
    def _pawn_structure_eval(self, board: chess.Board) -> float:
        """Evaluate pawn structure."""
        score = 0.0
        
        # Doubled pawns (penalty)
        for file in range(8):
            white_pawns = sum(1 for rank in range(8) 
                            if board.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, chess.WHITE))
            black_pawns = sum(1 for rank in range(8) 
                            if board.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, chess.BLACK))
            
            if white_pawns > 1:
                score -= 20 * (white_pawns - 1)  # Penalty for doubled pawns
            if black_pawns > 1:
                score += 20 * (black_pawns - 1)
        
        # Isolated pawns (penalty)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(square)
                is_isolated = True
                
                # Check adjacent files
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8:
                        for rank in range(8):
                            adj_square = chess.square(adj_file, rank)
                            adj_piece = board.piece_at(adj_square)
                            if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == piece.color:
                                is_isolated = False
                                break
                        if not is_isolated:
                            break
                
                if is_isolated:
                    if piece.color == chess.WHITE:
                        score -= 15
                    else:
                        score += 15
        
        return score
    
    def _king_safety_eval(self, board: chess.Board) -> float:
        """Evaluate king safety."""
        score = 0.0
        
        # Check if king is exposed (fewer pawns around)
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue
            
            rank = chess.square_rank(king_square)
            file = chess.square_file(king_square)
            
            # Count pawns around king
            pawn_shield = 0
            for dr in [-1, 0, 1]:
                for df in [-1, 0, 1]:
                    if dr == 0 and df == 0:
                        continue
                    r = rank + dr
                    f = file + df
                    if 0 <= r < 8 and 0 <= f < 8:
                        square = chess.square(f, r)
                        piece = board.piece_at(square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            pawn_shield += 1
            
            # Penalty for exposed king
            if pawn_shield < 2:
                if color == chess.WHITE:
                    score -= (2 - pawn_shield) * 20
                else:
                    score += (2 - pawn_shield) * 20
        
        return score
    
    def _mobility_eval(self, board: chess.Board) -> float:
        """Evaluate piece mobility."""
        score = 0.0
        
        # Count legal moves for each side
        board_copy = board.copy()
        
        # White mobility
        board_copy.turn = chess.WHITE
        white_moves = len(list(board_copy.legal_moves))
        
        # Black mobility
        board_copy.turn = chess.BLACK
        black_moves = len(list(board_copy.legal_moves))
        
        # Mobility bonus (more moves = better)
        score += (white_moves - black_moves) * 2
        
        return score

