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
    see_value = captured_value - capturing_value
    
    # CRITICAL IMPROVEMENT: Account for defenders
    # If the captured piece has no defenders, it's a FREE capture (much better)
    # If it has defenders, the exchange might be more complex
    defenders = board.attackers(not board.turn, move.to_square)
    attackers = board.attackers(board.turn, move.to_square)
    
    # Remove the capturing piece from attackers count (it's making the capture)
    attackers_count = len([sq for sq in attackers if sq != move.from_square])
    
    # If no defenders, this is a FREE capture - add bonus
    if len(defenders) == 0:
        # Free capture bonus: 50% of captured piece value
        see_value += captured_value * 0.5
    # If more attackers than defenders after capture, still good
    elif attackers_count > len(defenders):
        # Winning exchange - small bonus
        see_value += captured_value * 0.1
    
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

