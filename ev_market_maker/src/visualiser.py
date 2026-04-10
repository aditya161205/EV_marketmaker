
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional
 
from src.game_env import CardGameState
from src.ev_engine import compute_variance, compute_spread_width
from src.simulator import SimulationResult, GameResult
 
 
# consistent style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "lines.linewidth": 1.8,
    "font.family": "monospace",
})
 
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_ORANGE = "#d29922"
ACCENT_PURPLE = "#bc8cff"
 
 
# ------------------------------------------------------------------
# Chart 1: Spread width vs information revealed
# ------------------------------------------------------------------
 
def plot_spread_collapse(save_path: Optional[str] = None):
    """
    Theoretical spread width as cards are drawn from the deck.
    This should show a convex curve — wide at the start, collapsing near zero
    as the last few cards are revealed.
 
    This is the core visual that shows the agent thinks like a market maker.
    """
    state = CardGameState()
    state.shuffle(seed=99)
 
    spreads = []
    cards_remaining = []
    evs = []
 
    # draw cards one by one and record the spread at each step
    k = 0.5
    for _ in range(51):  # leave one card
        n_rem = state.cards_remaining()
        spread = compute_spread_width(state, k)
        spreads.append(spread)
        cards_remaining.append(n_rem)
        state.draw_card()
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spread Width vs Information Revealed", fontsize=14, color="#e6edf3", y=1.02)
 
    # left: spread width vs cards remaining
    ax1.plot(cards_remaining, spreads, color=ACCENT_BLUE, linewidth=2)
    ax1.fill_between(cards_remaining, spreads, alpha=0.15, color=ACCENT_BLUE)
    ax1.set_xlabel("Cards Remaining in Deck")
    ax1.set_ylabel("Spread Width")
    ax1.set_title("Spread Collapses as Deck Empties", color="#8b949e", fontsize=10)
    ax1.invert_xaxis()
    ax1.grid(True)
    ax1.axvline(x=10, color=ACCENT_ORANGE, linestyle=":", alpha=0.7, label="10 cards left")
    ax1.legend(framealpha=0.2)
 
    # right: spread vs cards DRAWN (mirrors left but shows progression)
    cards_drawn = [52 - r for r in cards_remaining]
    ax2.plot(cards_drawn, spreads, color=ACCENT_GREEN, linewidth=2)
    ax2.fill_between(cards_drawn, spreads, alpha=0.15, color=ACCENT_GREEN)
    ax2.set_xlabel("Cards Drawn So Far")
    ax2.set_ylabel("Spread Width")
    ax2.set_title("More Information → Tighter Spread", color="#8b949e", fontsize=10)
    ax2.grid(True)
 
    # annotate the starting and ending spread
    ax2.annotate(
        f"Start: {spreads[0]:.1f}",
        xy=(cards_drawn[0], spreads[0]),
        xytext=(5, spreads[0] - 5),
        color=ACCENT_ORANGE,
        fontsize=9,
    )
    ax2.annotate(
        f"End: {spreads[-1]:.2f}",
        xy=(cards_drawn[-1], spreads[-1]),
        xytext=(cards_drawn[-1] - 15, spreads[-1] + 3),
        color=ACCENT_ORANGE,
        fontsize=9,
    )
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()
 
 
# ------------------------------------------------------------------
# Chart 2: EV convergence on a single demo game
# ------------------------------------------------------------------
 
def plot_demo_game(result: GameResult, save_path: Optional[str] = None):
    """
    For a single demo game, plot:
    - The bid/ask band as cards are drawn
    - The true remaining sum (what the MM is actually trying to estimate)
    - The EV (mid of the band) converging to the true sum
    """
    snaps = result.snapshots
    if not snaps:
        print("No snapshots recorded — run with record_snapshots=True")
        return
 
    draws = [s["cards_drawn"] for s in snaps]
    evs = [s["ev"] for s in snaps]
    bids = [s["bid"] for s in snaps]
    asks = [s["ask"] for s in snaps]
    true_sums = [s["true_remaining_sum"] for s in snaps]
    spreads = [s["spread_width"] for s in snaps]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Live Demo: EV Convergence & Spread Dynamics", fontsize=13, color="#e6edf3")
 
    # top: bid/ask band + true sum
    ax1.fill_between(draws, bids, asks, alpha=0.2, color=ACCENT_BLUE, label="Bid-Ask Band")
    ax1.plot(draws, evs, color=ACCENT_BLUE, linewidth=2, label="EV (Mid)")
    ax1.plot(draws, true_sums, color=ACCENT_GREEN, linewidth=1.5,
             linestyle="--", label="True Remaining Sum")
    ax1.plot(draws, bids, color=ACCENT_BLUE, linewidth=0.8, alpha=0.5)
    ax1.plot(draws, asks, color=ACCENT_BLUE, linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Value")
    ax1.set_title("EV Tracks True Sum as Cards Are Revealed", color="#8b949e", fontsize=10)
    ax1.legend(framealpha=0.2, loc="upper right")
    ax1.grid(True)
 
    # bottom: spread width
    ax2.bar(draws, spreads, color=ACCENT_PURPLE, alpha=0.7, width=0.8)
    ax2.set_xlabel("Cards Drawn")
    ax2.set_ylabel("Spread Width")
    ax2.set_title("Spread Narrows With Each Card Drawn", color="#8b949e", fontsize=10)
    ax2.grid(True)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()
 
 
# ------------------------------------------------------------------
# Chart 3: P&L curve across N games
# ------------------------------------------------------------------
 
def plot_pnl_curve(result: SimulationResult, save_path: Optional[str] = None):
    """
    Cumulative P&L across all games, with:
    - Sharpe ratio annotation
    - Maximum drawdown highlighted
    - Rolling win rate
    """
    pnls = np.array([r.pnl for r in result.game_results])
    cumulative = np.cumsum(pnls)
    n = len(pnls)
    games = np.arange(1, n + 1)
 
    # compute running max and drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
 
    # rolling win rate (window=200)
    window = min(200, n // 10)
    rolling_wins = np.convolve(
        (pnls > 0).astype(float),
        np.ones(window) / window,
        mode="valid"
    )
 
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.05)
 
    # top: cumulative P&L
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(games, cumulative, color=ACCENT_GREEN, linewidth=1.5, label="Cumulative P&L")
    ax1.fill_between(games, cumulative, alpha=0.1, color=ACCENT_GREEN)
    ax1.plot(games, running_max, color=ACCENT_BLUE, linewidth=0.8,
             linestyle=":", alpha=0.6, label="Running Peak")
 
    # shade the max drawdown period
    max_dd_end = int(np.argmax(drawdowns))
    ax1.axvspan(
        max(0, max_dd_end - 200), max_dd_end,
        alpha=0.1, color=ACCENT_RED, label=f"Max DD Region"
    )
 
    # annotations
    ax1.annotate(
        f"Sharpe: {result.sharpe_ratio:.2f}\n"
        f"Max DD: {result.max_drawdown:.1f}\n"
        f"Win Rate: {result.win_rate:.1%}",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, color="#e6edf3",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", alpha=0.8, edgecolor="#30363d"),
    )
 
    ax1.set_title(
        f"P&L Across {n:,} Games | Total: {result.total_pnl:.0f} | "
        f"Avg {result.avg_trades_per_game:.1f} trades/game",
        color="#8b949e", fontsize=10
    )
    ax1.set_ylabel("Cumulative P&L")
    ax1.legend(framealpha=0.2, fontsize=9)
    ax1.grid(True)
    ax1.set_xticklabels([])
 
    # middle: drawdown
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(games, -drawdowns, alpha=0.6, color=ACCENT_RED)
    ax2.plot(games, -drawdowns, color=ACCENT_RED, linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True)
    ax2.set_xticklabels([])
 
    # bottom: rolling win rate
    ax3 = fig.add_subplot(gs[2])
    rolling_x = np.arange(window, n + 1)
    ax3.plot(rolling_x, rolling_wins, color=ACCENT_ORANGE, linewidth=1.2)
    ax3.axhline(y=0.5, color="#30363d", linestyle="--", alpha=0.8)
    ax3.set_ylabel(f"Win Rate\n({window}g rolling)")
    ax3.set_xlabel("Game Number")
    ax3.set_ylim(0.2, 0.8)
    ax3.grid(True)
 
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()
 
 
# ------------------------------------------------------------------
# Chart 4: Adversarial comparison
# ------------------------------------------------------------------
 
def plot_adversarial_comparison(
    normal_result: SimulationResult,
    adversarial_result: SimulationResult,
    save_path: Optional[str] = None,
):
    """
    Side by side: normal mixed flow vs dominated by hard informed traders.
    Shows how the MM degrades under adversarial flow.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Market Maker Performance: Normal vs Adversarial Flow", fontsize=13)
 
    for ax, result, label, color in zip(
        axes,
        [normal_result, adversarial_result],
        ["Normal Flow (60% Noise)", "Adversarial (80% Informed)"],
        [ACCENT_GREEN, ACCENT_RED],
    ):
        pnls = np.array([r.pnl for r in result.game_results])
        cumulative = np.cumsum(pnls)
        games = np.arange(1, len(pnls) + 1)
 
        ax.plot(games, cumulative, color=color, linewidth=1.5)
        ax.fill_between(games, cumulative, alpha=0.12, color=color)
 
        ax.annotate(
            f"Sharpe: {result.sharpe_ratio:.2f}\n"
            f"Win Rate: {result.win_rate:.1%}\n"
            f"Total P&L: {result.total_pnl:.0f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=9, color="#e6edf3", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", alpha=0.8, edgecolor="#30363d"),
        )
 
        ax.set_title(label, color="#8b949e", fontsize=10)
        ax.set_xlabel("Game Number")
        ax.set_ylabel("Cumulative P&L")
        ax.grid(True)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()
 
 
# ------------------------------------------------------------------
# Chart 5: Variance decay — theoretical curve
# ------------------------------------------------------------------
 
def plot_variance_decay(save_path: Optional[str] = None):
    """
    Shows the theoretical hypergeometric variance as a function of
    cards remaining. This is the mathematical foundation of the whole project.
    """
    state = CardGameState()
    state.shuffle(seed=1)
 
    variances = []
    cards_remaining = []
 
    for _ in range(51):
        v = compute_variance(state)
        variances.append(v)
        cards_remaining.append(state.cards_remaining())
        state.draw_card()
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    ax.plot(cards_remaining[::-1], list(reversed(variances)),
            color=ACCENT_PURPLE, linewidth=2, label="Hypergeometric Var")
    ax.fill_between(
        cards_remaining[::-1], list(reversed(variances)),
        alpha=0.15, color=ACCENT_PURPLE
    )
 
    ax.set_xlabel("Cards Drawn")
    ax.set_ylabel("Variance of Remaining Sum")
    ax.set_title(
        "Hypergeometric Variance Decay\n"
        "Var = n × σ² × (N-n)/(N-1)",
        color="#8b949e", fontsize=11
    )
    ax.grid(True)
    ax.legend(framealpha=0.2)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()