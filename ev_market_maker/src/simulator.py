"""
simulator.py
 
Runs the full simulation loop.
 
The game contract: MM quotes on the sum of cards that will remain
after N_STOP cards are drawn. Settlement = actual sum of those leftover cards.
 
Each step:
  1. Compute EV and quote (bid/ask) given cards drawn so far and draws remaining
  2. Traders respond — some random (noise), some with informational edge
  3. MM records fills, updates inventory
  4. Draw next card (information update)
  5. Repeat until N_STOP draws done
  6. Settle: inventory * true_remaining_sum + cash collected
 
This mirrors how Optiver thinks about market making:
  - Quote around fair value
  - Profit from the spread on noise traders
  - Protect against informed traders who know more than you
"""
 
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
 
from src.game_env import CardGameState
from src.ev_engine import compute_ev, compute_quote, get_state_snapshot
from src.market_maker import MarketMaker
from src.traders import MixedFlow
 
 
@dataclass
class GameResult:
    """All the data we track for a single completed game."""
    game_id: int
    pnl: float
    n_trades: int
    snapshots: List[dict] = field(default_factory=list)
 
 
@dataclass
class SimulationResult:
    """Aggregate stats across all games."""
    game_results: List[GameResult]
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trades_per_game: float
 
 
# ------------------------------------------------------------------
# Single game runner
# ------------------------------------------------------------------
 
def run_one_game(
    mm: MarketMaker,
    flow: MixedFlow,
    state: CardGameState,
    game_id: int,
    n_stop: int = 40,
    record_snapshots: bool = False,
) -> GameResult:
    """
    One complete game. The MM quotes the contract at each step.
 
    The contract value = sum of the cards that will remain after n_stop draws.
    At any point, with d draws left and R cards remaining:
        EV = (R - d) * mean(remaining)
 
    Settlement = true sum of the (52 - n_stop) cards not drawn.
    """
    state.reset()
    mm.reset()
    snapshots = []
 
    for step in range(n_stop):
        if state.is_finished():
            break
 
        # draws still to go (including this one)
        draws_left = n_stop - step
 
        bid, ask = mm.quote(state, n_to_draw=draws_left)
 
        action = flow.act(state, bid, ask)
 
        if action == "buy" and ask != float("inf"):
            mm.fill_sell(price=ask)   # counterparty buys = we sell at ask
        elif action == "sell" and bid != float("-inf"):
            mm.fill_buy(price=bid)    # counterparty sells = we buy at bid
 
        card = state.draw_card()
 
        if record_snapshots:
            draws_remaining_after = n_stop - step - 1
            snap = get_state_snapshot(state, mm.k, n_to_draw=max(0, draws_remaining_after))
            snap["bid"] = bid
            snap["ask"] = ask
            snap["inventory"] = mm.inventory
            snap["last_card"] = card
            snapshots.append(snap)
 
    # settlement against the true remaining sum
    true_remaining = state.true_remaining_sum()
    total_pnl = mm.settle(true_remaining)
 
    return GameResult(
        game_id=game_id,
        pnl=total_pnl,
        n_trades=len(mm.trades),
        snapshots=snapshots,
    )
 
 
# ------------------------------------------------------------------
# Multi-game simulation
# ------------------------------------------------------------------
 
def run_simulation(
    n_games: int = 10_000,
    k: float = 0.5,
    n_stop: int = 40,
    noise_weight: float = 0.6,
    soft_weight: float = 0.3,
    hard_weight: float = 0.1,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> SimulationResult:
    """
    Run the market maker against mixed flow across N independent games.
    """
    if seed is not None:
        np.random.seed(seed)
 
    mm = MarketMaker(k=k, inventory_limit=6, skew_factor=0.04, hit_penalty=1.25)
    flow = MixedFlow(noise_weight, soft_weight, hard_weight)
    state = CardGameState()
 
    results = []
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
 
    for i in range(n_games):
        result = run_one_game(mm, flow, state, game_id=i, n_stop=n_stop, record_snapshots=False)
        results.append(result)
        cumulative_pnl += result.pnl
 
        if cumulative_pnl > peak_pnl:
            peak_pnl = cumulative_pnl
        drawdown = peak_pnl - cumulative_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown
 
        if verbose and (i + 1) % 2000 == 0:
            print(f"  Game {i+1:,} | Cumulative P&L: {cumulative_pnl:.2f}")
 
    pnls = np.array([r.pnl for r in results])
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0.0
    win_rate = float(np.mean(pnls > 0))
    avg_trades = float(np.mean([r.n_trades for r in results]))
 
    return SimulationResult(
        game_results=results,
        total_pnl=float(pnls.sum()),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(max_drawdown),
        win_rate=win_rate,
        avg_trades_per_game=avg_trades,
    )
 
 
def run_demo_game(k: float = 0.5, n_stop: int = 38, seed: int = 7) -> GameResult:
    """
    Single game with full snapshots — for the live demo and visualisation.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
 
    mm = MarketMaker(k=k, inventory_limit=6)
    flow = MixedFlow(noise_weight=0.5, soft_informed_weight=0.35, hard_informed_weight=0.15)
    state = CardGameState()
 
    return run_one_game(mm, flow, state, game_id=0, n_stop=n_stop, record_snapshots=True)
 
 
def run_adversarial_test(n_games: int = 1000, k: float = 0.5) -> SimulationResult:
    """Stress test with mostly hard-informed traders."""
    return run_simulation(
        n_games=n_games,
        k=k,
        noise_weight=0.1,
        soft_weight=0.1,
        hard_weight=0.8,
        verbose=False,
    )
 