"""Board state utilities."""
import chess


def board_to_fen(board: chess.Board) -> str:
    """
    Convert chess.Board to FEN string.
    
    Args:
        board: chess.Board object
        
    Returns:
        FEN string
    """
    return board.fen()


def fen_to_board(fen: str) -> chess.Board:
    """
    Convert FEN string to chess.Board.
    
    Args:
        fen: FEN string
        
    Returns:
        chess.Board object
    """
    return chess.Board(fen)


def is_game_over(board: chess.Board) -> bool:
    """
    Check if game is over.
    
    Args:
        board: chess.Board object
        
    Returns:
        True if game is over (checkmate, stalemate, etc.)
    """
    return (
        board.is_checkmate() or
        board.is_stalemate() or
        board.is_insufficient_material() or
        board.is_seventyfive_moves() or
        board.is_fivefold_repetition()
    )

