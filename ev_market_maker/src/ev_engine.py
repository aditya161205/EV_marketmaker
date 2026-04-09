"""

The game setup: a 52-card deck is shuffled. The MM quotes a two-way market
on the CONTRACT: "what is the sum of the cards that remain at settlement?"
Settlement happens after exactly N cards are drawn (default N=40).
So the final payoff = sum of the last (52 - N) = 12 undrawn cards.
As each card is revealed, the MM updates its quote based on new information.
-----------
Let:
    R = cards currently remaining in the deck
    d = cards we will STILL draw before settlement  (= N_stop - cards_drawn_so_far)
    n_settle = R - d  = cards that will remain at settlement
E[settlement_sum | cards_seen] = n_settle * mean(remaining_pool)
Var[settlement_sum | cards_seen]
    = n_settle * var(remaining_pool) * d / (R - 1)
This is the hypergeometric variance: drawing d cards from R,
asking about the sum of the remaining (R - d).
Spread = k * sqrt(Var), which collapses to zero when d → 0
(no more draws, settlement sum is exactly known).
"""
 
import numpy as np
from typing import List, Tuple, Optional
 
from src.game_env import CardGameState, DiceGameState
 
 
# ------------------------------------------------------------------
# Card game EV engine
# ------------------------------------------------------------------
 
def compute_ev(state: CardGameState, n_to_draw: int = None) -> float:
    """
    E[settlement_sum | cards_seen_so_far].
 
    settlement_sum = sum of cards that will REMAIN after n_to_draw more are drawn.
 
    If n_to_draw >= len(remaining), the whole deck will be consumed and settlement = 0.
    If n_to_draw = 0, we settle right now and EV = sum(remaining).
    """
    remaining = state.remaining
    R = len(remaining)
 
    if R == 0:
        return 0.0
 
    if n_to_draw is None:
        # fallback: if no stop point specified, EV is the current remaining sum
        # (used for display / informal contexts only)
        return float(np.sum(remaining))
 
    d = min(n_to_draw, R)      # actual draws that will happen
    n_settle = R - d            # cards at settlement
 
    if n_settle <= 0:
        return 0.0
 
    return n_settle * float(np.mean(remaining))
 
 
def compute_variance(state: CardGameState, n_to_draw: int = None) -> float:
    """
    Var[settlement_sum | cards_seen_so_far].
 
    Hypergeometric variance: drawing d cards from R remaining,
    interested in the sum of the (R - d) cards NOT drawn.
 
        Var = n_settle * sigma^2(remaining) * d / (R - 1)
 
    This is zero when d=0 (settlement is certain) or d=R (all consumed, sum=0 always).
    It is maximised somewhere in between.
    """
    remaining = state.remaining
    R = len(remaining)
 
    if R <= 1 or n_to_draw is None:
        return 0.0
 
    d = min(n_to_draw, R - 1)  # draws before settlement (leave at least 1 card)
    n_settle = R - d
 
    if n_settle < 1:
        return 0.0
 
    pop_variance = float(np.var(remaining))
    if pop_variance < 1e-12:
        return 0.0
 
    return n_settle * pop_variance * d / (R - 1)
 
 
def compute_spread_width(state: CardGameState, k: float = 0.5, n_to_draw: int = None) -> float:
    """
    Spread width = k * sqrt(Var[settlement_sum]).
    Wide when uncertainty is high; collapses as draws approach the stop point.
    """
    var = compute_variance(state, n_to_draw=n_to_draw)
    return k * np.sqrt(var)
 
 
def compute_quote(
    state: CardGameState,
    k: float = 0.5,
    n_to_draw: int = None,
) -> Tuple[float, float]:
    """
    Returns (bid, ask) centered on EV with width driven by variance.
 
        bid  = EV - half_spread
        ask  = EV + half_spread
 
    As information arrives and variance collapses, the spread narrows
    and the mid converges to the true remaining sum.
    """
    ev = compute_ev(state, n_to_draw=n_to_draw)
    half_spread = compute_spread_width(state, k, n_to_draw=n_to_draw) / 2.0
 
    bid = ev - half_spread
    ask = ev + half_spread
 
    return round(bid, 4), round(ask, 4)
 
 
# ------------------------------------------------------------------
# Dice game EV engine (simpler case — independent rolls)
# ------------------------------------------------------------------
 
def compute_dice_ev(state: DiceGameState) -> float:
    """
    For dice, rolls are independent, so:
        E[remaining_sum] = dice_remaining * E[single_die]
    """
    return state.dice_remaining() * state.die_ev
 
 
def compute_dice_variance(state: DiceGameState) -> float:
    """
    Since dice rolls are independent:
        Var[remaining_sum] = dice_remaining * Var[single_die]
    """
    return state.dice_remaining() * state.die_var
 
 
def compute_dice_quote(state: DiceGameState, k: float = 0.5) -> Tuple[float, float]:
    ev = compute_dice_ev(state)
    half_spread = k * np.sqrt(compute_dice_variance(state)) / 2.0
    return round(ev - half_spread, 4), round(ev + half_spread, 4)
 
 
# ------------------------------------------------------------------
# Snapshot helper — used by the simulator to log state
# ------------------------------------------------------------------
 
def get_state_snapshot(
    state: CardGameState,
    k: float = 0.5,
    n_to_draw: int = None,
) -> dict:
    """
    Bundles all the numbers at a given game state into one dict.
    Used for logging and visualisation.
    """
    ev = compute_ev(state, n_to_draw=n_to_draw)
    var = compute_variance(state, n_to_draw=n_to_draw)
    std = np.sqrt(var) if var > 0 else 0.0
    bid, ask = compute_quote(state, k, n_to_draw=n_to_draw)
 
    return {
        "cards_drawn": state.cards_drawn(),
        "cards_remaining": state.cards_remaining(),
        "n_to_draw": n_to_draw,
        "ev": ev,
        "variance": var,
        "std": std,
        "spread_width": ask - bid,
        "bid": bid,
        "ask": ask,
        "true_remaining_sum": state.true_remaining_sum(),
    }