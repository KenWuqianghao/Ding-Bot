"""Static Exchange Evaluation (SEE) for better capture ordering."""
import chess


def piece_value(piece_type: int) -> int:
    """Get piece value for SEE calculation."""
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 10000
    }
    return values.get(piece_type, 0)


def see(board: chess.Board, move: chess.Move) -> int:
    """
    Static Exchange Evaluation - calculate actual capture value.
    
    Returns the net material gain from making this capture move.
    Positive = good capture, negative = bad capture.
    
    Args:
        board: chess.Board object
        move: Move to evaluate
        
    Returns:
        SEE value in centipawns
    """
    if not board.is_capture(move):
        return 0
    
    captured_piece = board.piece_at(move.to_square)
    if not captured_piece:
        return 0
    
    capturing_piece = board.piece_at(move.from_square)
    if not capturing_piece:
        return 0
    
    # Value of captured piece minus value of capturing piece
    captured_value = piece_value(captured_piece.piece_type)
    capturing_value = piece_value(capturing_piece.piece_type)
    
    # Basic SEE: captured - capturing
    # This is simplified - full SEE would simulate the exchange
    # For now, this is better than MVV-LVA because it accounts for attacker value
    see_value = captured_value - capturing_value
    
    # Bonus for promotion
    if move.promotion:
        see_value += piece_value(move.promotion) - piece_value(chess.PAWN)
    
    return see_value


def order_captures_by_see(board: chess.Board, captures: list) -> list:
    """
    Order captures by SEE value (highest first).
    
    Args:
        board: chess.Board object
        captures: List of capture moves
        
    Returns:
        List of captures ordered by SEE value (best first)
    """
    # Calculate SEE for each capture
    captures_with_see = [(move, see(board, move)) for move in captures]
    
    # Sort by SEE value (descending)
    captures_with_see.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the moves
    return [move for move, _ in captures_with_see]

