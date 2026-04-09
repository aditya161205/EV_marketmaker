"""
traders.py
 
Simulated counterparties for the market maker to trade against.
 
Two types:
  1. NoiseTrader — trades randomly with no informational edge.
     These are the MM's bread and butter — they pay the spread.
 
  2. InformedTrader — knows information the MM doesn't.
     Two variants:
       a. SoftInformed: has a slightly better estimate of fair value
       b. HardInformed: knows the NEXT card that will be drawn (adversarial)
"""
 
import random
import numpy as np
 
from src.game_env import CardGameState
from src.ev_engine import compute_ev
 
 
class NoiseTrader:
    """
    Hits bids and lifts asks with probability `trade_prob` each step.
    Has no view on fair value — purely random.
 
    This represents uninformed retail flow that keeps the MM's
    order book active between informed trades.
    """
 
    def __init__(self, trade_prob: float = 0.4):
        # probability of trading on any given quote
        self.trade_prob = trade_prob
 
    def act(self, bid: float, ask: float) -> str:
        """
        Returns 'buy' (lift ask), 'sell' (hit bid), or 'pass'.
        """
        if random.random() > self.trade_prob:
            return "pass"
 
        # roughly 50/50 between buying and selling
        return "buy" if random.random() < 0.5 else "sell"
 
 
class SoftInformedTrader:
    """
    Has a slightly better estimate of EV than the MM.
    Trades when the MM's mid deviates from their private estimate by
    more than `edge_threshold`.
 
    This simulates traders with a small informational advantage —
    they might have counted cards slightly better, or have access to
    slightly better probability estimates.
    """
 
    def __init__(self, edge_threshold: float = 1.5, trade_prob: float = 0.6):
        self.edge_threshold = edge_threshold
        self.trade_prob = trade_prob
 
    def act(self, state: CardGameState, bid: float, ask: float) -> str:
        """
        The informed trader knows the true EV of the remaining deck.
        If the MM's ask is below their EV estimate, they buy (it's cheap).
        If the MM's bid is above their EV estimate, they sell (it's expensive).
        """
        if random.random() > self.trade_prob:
            return "pass"
 
        # their "true" EV — in practice same formula but we assume they've
        # tracked the deck perfectly (no noise in their estimate)
        true_ev = compute_ev(state)
        mm_mid = (bid + ask) / 2.0
 
        if true_ev - mm_mid > self.edge_threshold and ask != np.inf:
            return "buy"   # MM is quoting too cheap, snap it up
        elif mm_mid - true_ev > self.edge_threshold and bid != -np.inf:
            return "sell"  # MM is quoting too rich, sell it
 
        return "pass"
 
 
class HardInformedTrader:
    """
    Knows the next card that will be drawn.
    Trades when knowing the next card would move the EV significantly.
 
    This is the adversarial case from Week 3 — it stress tests the MM's
    spread-widening mechanism. After consecutive hits from this trader,
    the MM should widen spreads to protect itself.
 
    Think of this as a trader who's peeked at the top of the deck.
    """
 
    def __init__(self, min_edge: float = 2.0, trade_prob: float = 0.8):
        self.min_edge = min_edge
        self.trade_prob = trade_prob
 
    def act(self, state: CardGameState, bid: float, ask: float) -> str:
        """
        Peeks at the next card and computes how EV shifts after it's revealed.
        If the shift is large enough, trades in the profitable direction.
        """
        if not state.remaining or random.random() > self.trade_prob:
            return "pass"
 
        next_card = state.remaining[0]  # informed: knows the next card
        current_ev = compute_ev(state)
 
        # simulate what EV would be after this card is revealed
        # (they trade BEFORE the card is flipped)
        remaining_after = state.remaining[1:]
        if not remaining_after:
            return "pass"
 
        ev_after = len(remaining_after) * np.mean(remaining_after)
        ev_shift = ev_after - current_ev
 
        mm_mid = (bid + ask) / 2.0
 
        # if the next card is a low card, EV drops after reveal — sell now
        # if it's a high card, EV rises after reveal — buy now
        if ev_shift > self.min_edge and ask != np.inf:
            return "buy"
        elif ev_shift < -self.min_edge and bid != -np.inf:
            return "sell"
 
        return "pass"
 
 
# ------------------------------------------------------------------
# Convenience: mixed flow (realistic scenario)
# ------------------------------------------------------------------
 
class MixedFlow:
    """
    Combines noise and informed traders into a realistic order flow.
    At each step, one of the traders randomly gets to act.
 
    Weights control how often each type shows up.
    """
 
    def __init__(
        self,
        noise_weight: float = 0.6,
        soft_informed_weight: float = 0.3,
        hard_informed_weight: float = 0.1,
    ):
        total = noise_weight + soft_informed_weight + hard_informed_weight
        self.weights = [
            noise_weight / total,
            soft_informed_weight / total,
            hard_informed_weight / total,
        ]
        self.noise = NoiseTrader(trade_prob=0.5)
        self.soft = SoftInformedTrader(edge_threshold=1.0, trade_prob=0.7)
        self.hard = HardInformedTrader(min_edge=1.5, trade_prob=0.8)
 
    def act(self, state: CardGameState, bid: float, ask: float) -> str:
        trader_type = random.choices(
            ["noise", "soft", "hard"], weights=self.weights
        )[0]
 
        if trader_type == "noise":
            return self.noise.act(bid, ask)
        elif trader_type == "soft":
            return self.soft.act(state, bid, ask)
        else:
            return self.hard.act(state, bid, ask)
 