"""
market_maker.py
The market making agent. It quotes two-way markets on the settlement contract
(sum of cards remaining after N draws), tracks inventory and cash P&L,
and adapts spread width based on inventory skew and informed flow detection.
"""
 
import numpy as np
from typing import Tuple, Optional
 
from src.game_env import CardGameState
from src.ev_engine import compute_quote, compute_ev
 
 
class MarketMaker:
 
    def __init__(
        self,
        k: float = 0.5,
        inventory_limit: int = 5,
        skew_factor: float = 0.05,
        hit_penalty: float = 1.3,
    ):
        self.k = k
        self.inventory_limit = inventory_limit
        self.skew_factor = skew_factor
        self.hit_penalty = hit_penalty
 
        self.inventory: int = 0
        self.cash: float = 0.0
        self.trades: list = []
        self.consecutive_hits: int = 0
        self.last_hit_side: Optional[str] = None
 
    def reset(self):
        self.inventory = 0
        self.cash = 0.0
        self.trades = []
        self.consecutive_hits = 0
        self.last_hit_side = None
 
    def quote(self, state: CardGameState, n_to_draw: int = None) -> Tuple[float, float]:
        """
        Generate a bid/ask centered on EV, adjusted for:
        1. Inventory skew — shade mid away from the overheld side
        2. Informed flow widening — after 3+ consecutive same-side hits
        """
        base_bid, base_ask = compute_quote(state, self.k, n_to_draw=n_to_draw)
        half_spread = (base_ask - base_bid) / 2.0
 
        # inventory skew: if long, shade bids/asks down to attract sellers
        skew = self.inventory * self.skew_factor * half_spread
        bid = base_bid - skew
        ask = base_ask - skew
 
        # widen spread if we're taking consecutive one-sided hits
        if self.consecutive_hits >= 3:
            penalty = self.hit_penalty ** (self.consecutive_hits - 2)
            extra = half_spread * (penalty - 1)
            bid -= extra
            ask += extra
 
        # hard inventory cap: don't quote the side that breaches the limit
        if self.inventory >= self.inventory_limit:
            bid = -np.inf
        if self.inventory <= -self.inventory_limit:
            ask = np.inf
 
        return round(bid, 4), round(ask, 4)
 
    def fill_buy(self, price: float, quantity: int = 1):
        """A counterparty sold to us (hit our bid). We are now longer."""
        self.inventory += quantity
        self.cash -= price * quantity
        self.trades.append({"side": "buy", "price": price, "qty": quantity})
        self._update_hit_counter("buy")
 
    def fill_sell(self, price: float, quantity: int = 1):
        """A counterparty bought from us (lifted our ask). We are now shorter."""
        self.inventory -= quantity
        self.cash += price * quantity
        self.trades.append({"side": "sell", "price": price, "qty": quantity})
        self._update_hit_counter("sell")
 
    def _update_hit_counter(self, side: str):
        if side == self.last_hit_side:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 1
            self.last_hit_side = side
 
    def settle(self, settlement_price: float) -> float:
        """
        End-of-game settlement. The contract pays the true remaining sum.
        Total P&L = cash from spread + inventory * settlement_price.
        """
        return self.cash + self.inventory * settlement_price
 
    def pnl_mark(self, mark_price: float) -> dict:
        mtm = self.cash + self.inventory * mark_price
        return {
            "cash": self.cash,
            "inventory": self.inventory,
            "mark": mark_price,
            "mtm_pnl": mtm,
            "n_trades": len(self.trades),
        }
 
    def __repr__(self):
        return (
            f"MarketMaker(inventory={self.inventory}, cash={self.cash:.2f}, "
            f"trades={len(self.trades)}, consecutive_hits={self.consecutive_hits})"
        )
 