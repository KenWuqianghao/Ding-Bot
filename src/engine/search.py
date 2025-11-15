"""Minimax search with alpha-beta pruning and neural network guidance."""
import chess
import time
from typing import Tuple, Optional, Dict, List
import math
from collections import defaultdict

from .evaluation import NNEvaluator
from .move_generation import get_legal_moves, order_moves_by_policy, order_moves_by_captures
from .zobrist import ZobristHash
from .see import see, order_captures_by_see


class MinimaxSearch:
    """
    Minimax search with alpha-beta pruning, guided by neural network.
    Includes transposition table, quiescence search, and advanced move ordering.
    """
    
    def __init__(
        self,
        evaluator: NNEvaluator,
        max_depth: int = 6,
        time_limit: Optional[float] = None
    ):
        """
        Initialize minimax search.
        
        Args:
            evaluator: NNEvaluator instance
            max_depth: Maximum search depth (default 6 for stronger play)
            time_limit: Optional time limit in seconds
        """
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = None
        self.nodes_searched = 0
        
        # Zobrist hashing for fast TT lookups
        self.zobrist = ZobristHash(seed=42)
        
        # Transposition table: hash -> (score, move, flag, depth)
        # Using hash instead of FEN for faster lookups
        self.transposition_table: Dict[int, Tuple[float, Optional[chess.Move], str, int]] = {}
        
        # Killer moves: moves that caused beta cutoffs at each depth
        self.killer_moves: Dict[int, List[chess.Move]] = defaultdict(list)
        
        # History heuristic: tracks how often moves cause cutoffs
        self.history: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Quiescence search depth
        self.quiescence_depth = 3
        
        # Search optimization flags
        # DISABLED aggressive pruning - may cause bad moves
        self.use_lmr = False  # Late Move Reductions - DISABLED
        self.use_razoring = False  # Razoring - DISABLED
        self.use_futility = False  # Futility pruning - DISABLED (was causing bad moves)
        self.use_delta = False  # Delta pruning - DISABLED
        self.use_check_extensions = True  # Check extensions - safe
        self.use_see = True  # Static Exchange Evaluation - safe
        
        # Extension tracking (prevent explosion)
        self.max_extensions = 2
    
    def order_moves(
        self,
        board: chess.Board,
        moves: List[chess.Move],
        depth: int,
        tt_move: Optional[chess.Move] = None
    ) -> List[chess.Move]:
        """
        Order moves using multiple heuristics for better alpha-beta pruning.
        
        Ordering priority:
        1. Transposition table move
        2. Captures (MVV-LVA)
        3. Checks
        4. Killer moves
        5. History heuristic
        6. Policy head (if available)
        """
        if not moves:
            return moves
        
        # Separate moves into categories
        tt_move_list = []
        captures = []
        checks = []
        killers = []
        history_moves = []
        quiet = []
        
        for move in moves:
            if move == tt_move:
                tt_move_list.append(move)
            elif board.is_capture(move):
                captures.append(move)
            elif board.gives_check(move):
                checks.append(move)
            elif move in self.killer_moves.get(depth, []):
                killers.append(move)
            elif (move.from_square, move.to_square) in self.history:
                history_moves.append(move)
            else:
                quiet.append(move)
        
        # Sort captures by SEE (Static Exchange Evaluation) if enabled, else MVV-LVA
        if self.use_see:
            captures = order_captures_by_see(board, captures)
        else:
            # MVV-LVA fallback
            piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                           chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
            
            def capture_value(move):
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if captured and attacker:
                    return piece_values.get(captured.piece_type, 0) - piece_values.get(attacker.piece_type, 0)
                return 0
            
            captures.sort(key=capture_value, reverse=True)
        
        # Sort history moves by history score
        history_moves.sort(key=lambda m: self.history.get((m.from_square, m.to_square), 0), reverse=True)
        
        # Try to use policy head for quiet moves
        try:
            _, policy_probs = self.evaluator.evaluate_with_policy(board)
            quiet.sort(key=lambda m: policy_probs.get(m, 0.0), reverse=True)
        except:
            pass
        
        # Combine in priority order
        ordered = tt_move_list + captures + checks + killers + history_moves + quiet
        return ordered
    
    def quiescence_search(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        depth: int = 0
    ) -> float:
        """
        Quiescence search to handle tactical positions.
        Only searches captures and checks to avoid horizon effect.
        Uses negamax framework.
        """
        if depth >= self.quiescence_depth:
            return self.evaluator.evaluate(board)
        
        # Stand pat (evaluator already returns from current player's perspective)
        stand_pat = self.evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        
        # Get captures and checks
        moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
        if not moves:
            return stand_pat
        
        # Order captures
        moves = order_moves_by_captures(board, moves)
        
        for move in moves:
            board.push(move)
            # Negamax: negate and swap bounds
            score = -self.quiescence_search(board, -beta, -alpha, depth + 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float = -math.inf,
        beta: float = math.inf,
        ply: int = 0
    ) -> Tuple[Optional[chess.Move], float]:
        """
        Negamax search with alpha-beta pruning, transposition table, and null move pruning.
        Uses negamax framework: always maximize from current player's perspective.
        
        Args:
            board: chess.Board object (will be modified with push/pop - should be a copy)
            depth: Remaining search depth
            alpha: Alpha value for alpha-beta pruning (lower bound)
            beta: Beta value for alpha-beta pruning (upper bound)
            ply: Current ply (distance from root)
            
        Returns:
            Tuple of (best_move, score) from current player's perspective
        """
        # Check time limit (only at root or after significant work)
        # Don't check on every recursive call - too expensive
        if self.time_limit and self.start_time and ply == 0:
            elapsed = time.time() - self.start_time
            if elapsed >= self.time_limit * 0.95:  # Use 95% of time
                score = self.evaluator.evaluate(board)
                return None, score
        
        self.nodes_searched += 1
        
        # Check transposition table using Zobrist hash (faster than FEN)
        position_hash = self.zobrist.hash_board(board)
        tt_entry = self.transposition_table.get(position_hash)
        if tt_entry:
            tt_score, tt_move, tt_flag, tt_depth = tt_entry
            # Only use TT entry if it was searched to at least the same depth
            if tt_depth >= depth:
                # Use TT entry if exact score or if it gives us a cutoff
                if tt_flag == 'exact':
                    return tt_move, tt_score
                elif tt_flag == 'lowerbound' and tt_score >= beta:
                    return tt_move, tt_score
                elif tt_flag == 'upperbound' and tt_score <= alpha:
                    return tt_move, tt_score
        
        # Terminal conditions
        if board.is_checkmate():
            score = -10000 + ply  # Prefer shorter mates (from current player's perspective)
            return None, score
        
        # Mate distance pruning: if alpha is already a mate score and we can't improve mate distance, prune
        if alpha > 9000:
            mate_distance = 10000 - alpha
            if ply >= mate_distance:
                return None, alpha
        
        if board.is_stalemate() or board.is_insufficient_material():
            return None, 0.0
        if board.is_repetition(3):
            return None, 0.0
        
        # Check extensions: extend search when in check (critical positions)
        extension = 0
        if self.use_check_extensions and board.is_check() and depth > 0:
            extension = 1
        
        # Adjust depth with extension
        adjusted_depth = depth + extension
        
        # Quiescence search at leaf nodes
        if adjusted_depth == 0:
            score = self.quiescence_search(board, alpha, beta)
            return None, score
        
        # Razoring: reduce depth for quiet positions with low static evaluation
        if self.use_razoring and adjusted_depth >= 4 and not board.is_check():
            static_eval = self.evaluator.evaluate(board)
            margin = 200 * adjusted_depth  # Margin increases with depth
            if static_eval + margin < alpha:
                # Razor: reduce depth by 2
                adjusted_depth = max(1, adjusted_depth - 2)
        
        # Null move pruning (skip if in check or endgame)
        # DISABLED - may have bugs causing early returns
        # if depth >= 3 and not board.is_check():
        #     # Check if we have enough material (skip in endgame)
        #     material_count = sum(1 for square in chess.SQUARES if board.piece_at(square))
        #     if material_count > 10:  # Not endgame
        #         board.push(chess.Move.null())
        #         # Negamax: negate score and swap bounds
        #         null_score = -self.search(board, depth - 1 - 2, -beta, -beta + 1, ply + 1)[1]
        #         board.pop()
        #         
        #         if null_score >= beta:
        #             return None, beta  # Beta cutoff
        
        # Get legal moves
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            score = self.evaluator.evaluate(board)
            return None, score
        
        # Get TT move if available
        tt_move = tt_entry[1] if tt_entry else None
        
        # Order moves
        legal_moves = self.order_moves(board, legal_moves, depth, tt_move)
        
        best_move = None
        best_score = -math.inf
        original_alpha = alpha  # Save original alpha for flag determination
        
        # Get static evaluation for futility and delta pruning
        static_eval = self.evaluator.evaluate(board) if (self.use_futility or self.use_delta) else None
        
        for move_idx, move in enumerate(legal_moves):
            # Delta pruning: skip captures that can't improve alpha
            if self.use_delta and board.is_capture(move) and static_eval is not None:
                capture_value = see(board, move)
                margin = 200  # Safety margin
                if static_eval + capture_value + margin < alpha:
                    continue  # Skip this capture
            
            # Futility pruning: skip quiet moves unlikely to improve alpha
            if self.use_futility and adjusted_depth >= 3 and not board.is_capture(move) and not board.gives_check(move):
                if static_eval is not None:
                    margin = 150 * adjusted_depth  # Margin increases with depth
                    if static_eval + margin < alpha:
                        continue  # Skip this quiet move
            
            # Make move
            board.push(move)
            
            # Late Move Reduction: reduce depth for quiet moves later in ordering
            reduction = 0
            if self.use_lmr and move_idx >= 3 and adjusted_depth >= 3:
                # Only reduce quiet moves (not captures, checks, promotions)
                if not board.is_check() and not board.is_capture(move) and not board.gives_check(move) and move.promotion is None:
                    # Progressive reduction: more reduction for later moves
                    if move_idx < 6:
                        reduction = 1
                    else:
                        reduction = 2
            
            search_depth = adjusted_depth - 1 - reduction
            
            # Principal Variation Search (PVS)
            # First move: full window search
            # Subsequent moves: null window search (faster), then re-search if needed
            if move_idx == 0:
                # First move: full window search
                _, score = self.search(
                    board,
                    search_depth,
                    -beta,  # Negate and swap for negamax
                    -alpha,  # Negate and swap for negamax
                    ply + 1
                )
                score = -score  # Negate score from opponent's perspective
            else:
                # Subsequent moves: null window search first (faster)
                # Search with null window [-alpha-1, -alpha] to quickly check if move beats alpha
                _, score = self.search(
                    board,
                    search_depth,
                    -alpha - 1,  # Null window lower bound
                    -alpha,  # Null window upper bound (just above current alpha)
                    ply + 1
                )
                score = -score  # Negate score from opponent's perspective
                
                # If null window search shows score > alpha, re-search with full window
                # This happens when the move is actually good (beats alpha)
                if score > alpha:
                    # Re-search with full window to get accurate score
                    _, score = self.search(
                        board,
                        search_depth,
                        -beta,  # Full window lower bound
                        -alpha,  # Full window upper bound
                        ply + 1
                    )
                    score = -score  # Negate score from opponent's perspective
            
            # Undo move
            board.pop()
            
            # Update best move (always maximizing in negamax)
            if score > best_score:
                best_score = score
                best_move = move
            
            # Update alpha (lower bound)
            if score > alpha:
                alpha = score
            
            # Alpha-beta pruning: if score >= beta, we have a cutoff
            if score >= beta:
                # Update killer moves
                if move not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, move)
                    if len(self.killer_moves[depth]) > 2:
                        self.killer_moves[depth].pop()
                
                # Update history heuristic
                self.history[(move.from_square, move.to_square)] += depth * depth
                
                # Beta cutoff: score is too good, opponent won't allow this
                best_score = beta
                best_move = move
                flag = 'lowerbound'
                break
        
        # Determine flag for transposition table (compare against original alpha)
        if best_score <= original_alpha:  # All moves were <= original alpha (fail low)
            flag = 'upperbound'
        elif best_score >= beta:  # We had a beta cutoff (fail high)
            flag = 'lowerbound'
        else:  # original_alpha < best_score < beta (exact)
            flag = 'exact'
        
        # Store in transposition table using Zobrist hash (faster than FEN)
        self.transposition_table[position_hash] = (best_score, best_move, flag, depth)
        
        return best_move, best_score
    
    def iterative_deepening(
        self,
        board: chess.Board,
        time_limit: Optional[float] = None
    ) -> chess.Move:
        """
        Iterative deepening search with time limit and improved time management.
        
        Args:
            board: chess.Board object
            time_limit: Time limit in seconds
            
        Returns:
            Best move found
        """
        # CRITICAL: Copy board to prevent state corruption
        # The search() function modifies the board with push/pop, but if there's
        # an exception or bug, the board state could be corrupted
        board = board.copy()
        
        if time_limit is not None:
            self.time_limit = time_limit
        self.start_time = time.time()
        self.nodes_searched = 0
        
        # Clear transposition table and heuristics for new search
        self.transposition_table.clear()
        self.killer_moves.clear()
        self.history.clear()
        
        best_move = None
        best_score = -math.inf
        
        # Aspiration window: start with narrow window around previous score
        aspiration_window = 50  # centipawns
        alpha_window = -math.inf
        beta_window = math.inf
        
        # Start with depth 1 and increase
        # For very short time limits, reduce max depth to ensure we complete at least depth 2-3
        effective_max_depth = self.max_depth
        if self.time_limit and self.time_limit < 1.0:
            # For <1s, limit to depth 4-5 max to ensure completion
            effective_max_depth = min(self.max_depth, 5)
        
        for depth in range(1, effective_max_depth + 1):
            # Check if time is up (check more frequently for short time controls)
            if self.time_limit is not None:
                elapsed = time.time() - self.start_time
                
                # For very short time limits (< 2s), be more aggressive
                if self.time_limit < 2.0:
                    # Use 90% of time, save 10% for safety
                    if elapsed >= self.time_limit * 0.90:
                        break
                    # Stop early if we've done at least depth 2 and time is running out
                    if depth >= 2 and elapsed > self.time_limit * 0.70:
                        break
                else:
                    # Normal time management: use 95% of time
                    if elapsed >= self.time_limit * 0.95:
                        break
                
                # Estimate time for next depth (rough heuristic)
                # For short time controls, be more aggressive about stopping
                if depth >= 3 and elapsed > 0.5:  # Need at least 0.5s of data
                    time_per_depth = elapsed / depth
                    estimated_next = time_per_depth * (depth + 1)
                    # Stop if estimated time would exceed limit
                    if self.time_limit < 2.0:
                        # Very aggressive for short time
                        if estimated_next > self.time_limit * 0.85:
                            break
                    else:
                        # Normal threshold
                        if estimated_next > self.time_limit * 0.95:
                            break
            
            # Aspiration window: use narrow window if we have a previous score
            if depth >= 3 and best_score != -math.inf:
                alpha_window = best_score - aspiration_window
                beta_window = best_score + aspiration_window
            else:
                alpha_window = -math.inf
                beta_window = math.inf
            
            # Search at current depth with aspiration window
            move, score = self.search(
                board,
                depth,
                alpha=alpha_window,
                beta=beta_window,
                ply=0
            )
            
            # If aspiration window failed, re-search with full window
            # Check if score is outside the aspiration window bounds
            if move is None or score == float('inf') or score == -float('inf') or \
               (alpha_window != -math.inf and score <= alpha_window) or \
               (beta_window != math.inf and score >= beta_window):
                # Re-search with full window
                move, score = self.search(
                    board,
                    depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    ply=0
                )
            
            if move is None:
                break
            
            best_move = move
            best_score = score
            final_depth = depth  # Track final depth reached
            
            # Mate distance pruning: if we found a mate, stop searching
            if abs(score) > 9000:
                break
        
        # Fallback to first legal move if no move found
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
            else:
                return None
        
        # Debug output
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Search: depth={final_depth}, nodes={self.nodes_searched}, time={elapsed:.2f}s, move={best_move.uci() if best_move else 'none'}")
        
        return best_move

