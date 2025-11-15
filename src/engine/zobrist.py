"""Zobrist hashing for fast position hashing."""
import random
import chess


class ZobristHash:
    """
    Zobrist hashing for chess positions.
    Provides fast, incremental hash updates for transposition tables.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize Zobrist hash tables.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        
        # Hash table: [piece_type][color][square]
        # piece_type: 0=PAWN, 1=KNIGHT, 2=BISHOP, 3=ROOK, 4=QUEEN, 5=KING
        # color: 0=WHITE, 1=BLACK
        # square: 0-63
        self.piece_table = [
            [
                [self.rng.getrandbits(64) for _ in range(64)]
                for _ in range(2)
            ]
            for _ in range(6)
        ]
        
        # Castling rights: [white_kingside, white_queenside, black_kingside, black_queenside]
        self.castling = [self.rng.getrandbits(64) for _ in range(4)]
        
        # En passant file (0-7, or 8 if none)
        self.en_passant = [self.rng.getrandbits(64) for _ in range(9)]
        
        # Side to move (white=0, black=1)
        self.turn = self.rng.getrandbits(64)
    
    def hash_board(self, board: chess.Board) -> int:
        """
        Compute Zobrist hash for a chess board position.
        
        Args:
            board: chess.Board object
            
        Returns:
            64-bit hash value
        """
        h = 0
        
        # Hash pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type_idx = piece.piece_type - 1  # 0-5
                color_idx = 0 if piece.color == chess.WHITE else 1
                h ^= self.piece_table[piece_type_idx][color_idx][square]
        
        # Hash castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self.castling[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self.castling[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self.castling[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self.castling[3]
        
        # Hash en passant
        if board.ep_square:
            ep_file = chess.square_file(board.ep_square)
            h ^= self.en_passant[ep_file]
        else:
            h ^= self.en_passant[8]  # No en passant
        
        # Hash side to move
        if board.turn == chess.BLACK:
            h ^= self.turn
        
        return h
    
    def hash_move_update(self, old_hash: int, board: chess.Board, move: chess.Move) -> int:
        """
        Incrementally update hash after a move.
        More efficient than recomputing from scratch.
        
        Args:
            old_hash: Previous hash value
            board: Board before move
            move: Move to apply
            
        Returns:
            Updated hash value
        """
        h = old_hash
        
        # Get piece info before move
        from_piece = board.piece_at(move.from_square)
        to_piece = board.piece_at(move.to_square)
        
        if not from_piece:
            return old_hash  # Invalid move
        
        piece_type_idx = from_piece.piece_type - 1
        color_idx = 0 if from_piece.color == chess.WHITE else 1
        
        # Remove piece from source square
        h ^= self.piece_table[piece_type_idx][color_idx][move.from_square]
        
        # Handle capture
        if to_piece:
            captured_type_idx = to_piece.piece_type - 1
            captured_color_idx = 0 if to_piece.color == chess.WHITE else 1
            h ^= self.piece_table[captured_type_idx][captured_color_idx][move.to_square]
        
        # Handle promotion
        if move.promotion:
            promoted_type_idx = move.promotion - 1
            h ^= self.piece_table[promoted_type_idx][color_idx][move.to_square]
        else:
            # Place piece on destination square
            h ^= self.piece_table[piece_type_idx][color_idx][move.to_square]
        
        # Handle castling
        if board.is_castling(move):
            if move.to_square == chess.G1:  # White kingside
                h ^= self.piece_table[3][0][chess.H1]  # Remove rook
                h ^= self.piece_table[3][0][chess.F1]  # Add rook
            elif move.to_square == chess.C1:  # White queenside
                h ^= self.piece_table[3][0][chess.A1]  # Remove rook
                h ^= self.piece_table[3][0][chess.D1]  # Add rook
            elif move.to_square == chess.G8:  # Black kingside
                h ^= self.piece_table[3][1][chess.H8]  # Remove rook
                h ^= self.piece_table[3][1][chess.F8]  # Add rook
            elif move.to_square == chess.C8:  # Black queenside
                h ^= self.piece_table[3][1][chess.A8]  # Remove rook
                h ^= self.piece_table[3][1][chess.D8]  # Add rook
        
        # Update castling rights (simplified - just recompute)
        # For efficiency, we could track this incrementally too
        board.push(move)
        h ^= self.turn  # Toggle side to move
        
        # Update castling rights
        if not board.has_kingside_castling_rights(chess.WHITE):
            h ^= self.castling[0]
        if not board.has_queenside_castling_rights(chess.WHITE):
            h ^= self.castling[1]
        if not board.has_kingside_castling_rights(chess.BLACK):
            h ^= self.castling[2]
        if not board.has_queenside_castling_rights(chess.BLACK):
            h ^= self.castling[3]
        
        # Update en passant
        if board.ep_square:
            ep_file = chess.square_file(board.ep_square)
            h ^= self.en_passant[ep_file]
        else:
            h ^= self.en_passant[8]
        
        board.pop()
        
        return h

