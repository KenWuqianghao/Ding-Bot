"""Board state preprocessing and data augmentation utilities."""
import chess
import torch
import numpy as np
from typing import Tuple, Optional, Dict, List


def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Convert FEN string to 18-channel tensor representation.
    
    Channels:
    - 0-11: Piece channels (6 piece types × 2 colors)
    - 12: Side to move (1 for white, 0 for black)
    - 13-16: Castling rights (K, Q, k, q)
    - 17: En passant square (1 if available, 0 otherwise)
    
    Args:
        fen: FEN string representation of board
        
    Returns:
        Tensor of shape [18, 8, 8] with dtype float32
    """
    board = chess.Board(fen)
    tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
    
    # Piece channels (0-11)
    piece_channels = {
        chess.PAWN: {chess.WHITE: 0, chess.BLACK: 6},
        chess.ROOK: {chess.WHITE: 1, chess.BLACK: 7},
        chess.KNIGHT: {chess.WHITE: 2, chess.BLACK: 8},
        chess.BISHOP: {chess.WHITE: 3, chess.BLACK: 9},
        chess.QUEEN: {chess.WHITE: 4, chess.BLACK: 10},
        chess.KING: {chess.WHITE: 5, chess.BLACK: 11},
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)  # Convert to 0-7 row index
            col = square % 8
            channel = piece_channels[piece.piece_type][piece.color]
            tensor[channel, row, col] = 1.0
    
    # Side to move (channel 12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    
    # Castling rights (channels 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0
    
    # En passant (channel 17)
    if board.ep_square is not None:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        tensor[17, row, col] = 1.0
    
    return tensor


def augment_position(board_tensor: torch.Tensor, label: Dict) -> Tuple[torch.Tensor, Dict]:
    """
    Apply random augmentation to board position.
    
    Augmentations:
    - Horizontal flip (mirror board)
    - Color flip (swap white/black pieces)
    
    Args:
        board_tensor: Tensor of shape [18, 8, 8]
        label: Dictionary with 'value' and optionally 'policy' keys
        
    Returns:
        Augmented tensor and adjusted labels
    """
    import random
    
    augmented_tensor = board_tensor.clone()
    augmented_label = label.copy()
    
    # Random horizontal flip
    if random.random() < 0.5:
        # Flip all channels horizontally
        augmented_tensor = torch.flip(augmented_tensor, dims=[2])
        # Note: Policy labels would need to be adjusted, but for simplicity
        # we'll handle this at the dataset level if needed
    
    # Random color flip
    if random.random() < 0.5:
        # Swap white and black piece channels
        white_pieces = augmented_tensor[0:6, :, :].clone()
        black_pieces = augmented_tensor[6:12, :, :].clone()
        augmented_tensor[0:6, :, :] = black_pieces
        augmented_tensor[6:12, :, :] = white_pieces
        
        # Flip side to move
        augmented_tensor[12, :, :] = 1.0 - augmented_tensor[12, :, :]
        
        # Swap castling rights
        white_k = augmented_tensor[13, :, :].clone()
        white_q = augmented_tensor[14, :, :].clone()
        black_k = augmented_tensor[15, :, :].clone()
        black_q = augmented_tensor[16, :, :].clone()
        augmented_tensor[13, :, :] = black_k
        augmented_tensor[14, :, :] = black_q
        augmented_tensor[15, :, :] = white_k
        augmented_tensor[16, :, :] = white_q
        
        # Negate evaluation (perspective flip)
        if 'value' in augmented_label:
            augmented_label['value'] = -augmented_label['value']
    
    return augmented_tensor, augmented_label


def move_to_index(move: chess.Move, legal_moves: List[chess.Move]) -> int:
    """
    Convert chess.Move to index in legal moves list.
    
    Args:
        move: chess.Move object
        legal_moves: List of legal moves
        
    Returns:
        Index in legal_moves list, or -1 if move not legal
    """
    try:
        return legal_moves.index(move)
    except ValueError:
        return -1


def index_to_move(index: int, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
    """
    Convert index back to chess.Move.
    
    Args:
        index: Index in legal_moves list
        legal_moves: List of legal moves
        
    Returns:
        chess.Move object, or None if index is invalid
    """
    if 0 <= index < len(legal_moves):
        return legal_moves[index]
    return None


def move_to_uci(move: chess.Move) -> str:
    """Convert chess.Move to UCI string."""
    return move.uci()


def uci_to_move(uci_str: str, board: chess.Board) -> Optional[chess.Move]:
    """Convert UCI string to chess.Move, validating legality."""
    try:
        move = chess.Move.from_uci(uci_str)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    return None

