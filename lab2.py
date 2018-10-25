# AI Lab 2: Games and ConnectFour 

# Name(s): Aidan Glickman, Lennon Okun
# Email(s): aidgli20@bergen.org, lenoku20@bergen.org

from game_api import *
from boards import *
from toytree import GAME1
from time import time
import numpy as np
import math
INF = float('inf')


H = 6
W = 7

ROW_RANGE = list(range(H))
COL_RANGE = list(range(W))
# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

# TODO: maybe just have to do as he expected bc his api runs pretty deep and everything uses it and its rly icky
def is_game_over_connectfour(board):
    # Returns True if game is over, otherwise False.
    return any(len(chain) >= 4 for chain in board.get_all_chains()) \
    or all(board.is_column_full(col) for col in COL_RANGE)

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    return list(board.add_piece(col) for col in COL_RANGE
                if not board.is_column_full(col)) \
            if not is_game_over_connectfour(board) else []

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
        -1000 if the minimizer has won, or 0 in case of a tie."""
    #TODO
    if next((chain for chain in board.get_all_chains() if len(chain) >= 4), None):
        return -1000 if is_current_player_maximizer else 1000
    else:
        return 0



def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
        returning larger absolute scores for winning sooner."""
    return endgame_score_connectfour(board, is_current_player_maximizer) * (math.e ** -board.count_pieces() + 1)

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :

    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    stats = [0, None]
    path = dfsmax(state, stats)
    score = stats[1].get_endgame_score()

    return (path, score, stats[0])


def dfsmax(state, stats):
    if state.is_game_over():
        stats[0] += 1
        if not stats[1] or state.get_endgame_score() > stats[1].get_endgame_score():
            stats[1] = state

        return [state]

    to_add =  max((dfsmax(child, stats) for child in state.generate_next_states()), key = lambda x: x[-1].get_endgame_score())
    return [state] + to_add 


# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))

def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Returns the same as dfs_maximizing:
    a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    stats = [0]
    path, score = minimax(state, stats, maximize)
    return (path, score, stats[0])

def minimax(state, stats, maximize):
    if state.is_game_over():
        stats[0] += 1
        return [state], state.get_endgame_score(maximize)
    if maximize:
        path, value = max((minimax(child, stats, False) for child in state.generate_next_states()),
            key = lambda x: x[1])
        return [state] + path, value
    else:
        value = 1000
        path, value = min((minimax(child, stats, True) for child in state.generate_next_states()),
            key = lambda x: x[1])
        return [state] + path, value
        
# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

#pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


#### Part 3: Cutting off and Pruning search #############################################


def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    # TODO: maybe subtract dead chains

    heur = 0
    player1_chains = board.get_all_chains(True)    
    player2_chains = board.get_all_chains(False)

    for p1_chain in player1_chains:
        heur += len(p1_chain)**2
    for p2_chain in player2_chains:
        heur -= len(p2_chain)**2

    return heur if is_current_player_maximizer else -heur


## Note that the signature of heuristic_fn is heuristic_fn(board, maximize=True)

def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs h-minimax, cutting off search at depth_limit and using heuristic_fn
    to evaluate non-terminal states. 
    Same return type as dfs_maximizing, a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""

    stats = [0]
    path, score = hminimax(state, heuristic_fn, depth_limit, stats, maximize)
    return (path, score, stats[0])

def hminimax(state, heuristic_fn, depth_limit, stats, maximize):
    if state.is_game_over():
        stats[0] += 1
        return [state], state.get_endgame_score(maximize)
    if depth_limit == 0: # TODO: maybe 1
        stats[0] += 1
        return [state], heuristic_fn(state.snapshot, maximize)
    if maximize:
        path, value = max((hminimax(child, heuristic_fn, depth_limit-1, stats, False) for child in state.generate_next_states()),
            key = lambda x: x[1])
        return [state] + path, value
    else:
        value = 1000
        path, value = min((hminimax(child, heuristic_fn, depth_limit-1, stats, True) for child in state.generate_next_states()),
            key = lambda x: x[1])
        return [state] + path, value

# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=2))

def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. 
    Same return type as dfs_maximizing, a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    stats = [0]
    path, score = abminimax(state, alpha, beta, heuristic_fn, depth_limit, stats, maximize)
    return (path, score, stats[0])

def abminimax(state, alpha, beta, heuristic_fn, depth_limit, stats, maximize):
    if state.is_game_over():
        stats[0] += 1
        return [state], state.get_endgame_score(maximize)
    if depth_limit == 0: # TODO: maybe 1
        stats[0] += 1
        return [state], heuristic_fn(state.snapshot, maximize)

    children = state.generate_next_states()

    if maximize:
        val = -INF
        path = None
        for child in children:
            path2, val2 = abminimax(child, alpha, beta, heuristic_fn, depth_limit-1, stats, False)
            if val2 > val:
                val, path = val2, path2
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return [state] + path, val
    else:
        val = INF
        path = None
        for child in children:
            path2, val2 = abminimax(child, alpha, beta, heuristic_fn, depth_limit-1, stats, True)
            if val2 < val:
                val, path = val2, path2
            beta = min(beta, val)
            if alpha >= beta:
                break
        return [state] + path, val


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True, time_limit=INF) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. 
    Returns anytime_value."""
    raise NotImplementedError


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented




#
# If you want to enter the tournament, implement your final contestant 
# in this function. You may write other helper functions/classes 
# but the function must take these arguments (though it can certainly ignore them)
# and must return an AnytimeValue.
#
def tournament_search(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True, time_limit=INF) :
    """Runs some kind of search (probably progressive deepening).
    Returns an AnytimeValue."""
    raise NotImplementedError

