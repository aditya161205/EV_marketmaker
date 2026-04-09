
import argparse
import time
import numpy as np
 
from src.game_env import CardGameState
from src.ev_engine import compute_ev, compute_quote
from src.market_maker import MarketMaker
from src.traders import MixedFlow
from src.simulator import run_simulation, run_demo_game, run_adversarial_test
 
 
N_STOP   = 40           # number of cards drawn before settlement
N_SETTLE = 52 - N_STOP  # cards left at settlement = 12
 
 
# ------------------------------------------------------------------
# live demo
# ------------------------------------------------------------------
 
def run_narrated_demo(show_steps: int = 20):
    """
    Runs all 40 draws. Prints the first `show_steps` in detail,
    then silently finishes the game and settles.
    """
    print("\n" + "=" * 72)
    print("  LIVE DEMO: Continuous EV Market Maker on a Card Game")
    print("=" * 72)
    print(f"\n  Contract : sum of the {N_SETTLE} cards remaining after {N_STOP} draws")
    print(f"  Start EV : {N_SETTLE} x (340/52) = {N_SETTLE * 340/52:.2f}")
    print(f"  Spread   : k * sqrt(Var[settlement_sum | cards_seen])\n")
 
    hdr = f"  {'Step':<5} {'Card':<5} {'Rem':<5} {'Draws Left':<11} {'EV':>8} {'Bid':>9} {'Ask':>9} {'Spread':>8}  Action"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
 
    state = CardGameState()
    state.shuffle(seed=42)
    mm = MarketMaker(k=0.5)
    flow = MixedFlow()
 
    for step in range(N_STOP):
        draws_left = N_STOP - step  # including this draw
 
        bid, ask = mm.quote(state, n_to_draw=draws_left)
        ev       = compute_ev(state, n_to_draw=draws_left)
        spread   = (ask - bid) if (ask != float("inf") and bid != float("-inf")) else float("inf")
 
        action = flow.act(state, bid, ask)
        trade_str = ""
        if action == "buy" and ask != float("inf"):
            mm.fill_sell(price=ask)
            trade_str = "<-- SOLD (trader lifted ask)"
        elif action == "sell" and bid != float("-inf"):
            mm.fill_buy(price=bid)
            trade_str = "<-- BOUGHT (trader hit bid)"
 
        card = state.draw_card()
 
        if step < show_steps:
            sp_str = f"{spread:>8.3f}" if spread != float("inf") else f"{'inf':>8}"
            print(
                f"  {step+1:<5} {card:<5} {state.cards_remaining():<5} {draws_left-1:<11}"
                f" {ev:>8.2f} {bid:>9.2f} {ask:>9.2f} {sp_str}  {trade_str}"
            )
            time.sleep(0.03)
        elif step == show_steps:
            print(f"\n  ... ({N_STOP - show_steps} draws remaining, running silently) ...\n")
 
    # settlement: true sum of the 12 remaining cards
    true_remaining = state.true_remaining_sum()
    final_pnl      = mm.settle(true_remaining)
 
    print(f"  {'─' * 55}")
    print(f"  True settlement sum (12 cards left) : {true_remaining}")
    print(f"  Inventory at settlement             : {mm.inventory:+d}")
    print(f"  Cash collected from spread          : {mm.cash:+.2f}")
    print(f"  Inventory P&L (inv x {true_remaining})           : {mm.inventory * true_remaining:+.2f}")
    print(f"  {'─' * 55}")
    print(f"  TOTAL P&L                           : {final_pnl:+.2f}")
    print(f"  Total trades executed               : {len(mm.trades)}")
    print()
 
 
# ------------------------------------------------------------------
# Summary printer
# ------------------------------------------------------------------
 
def print_summary(result, label: str):
    print(f"\n{'━' * 50}")
    print(f"  {label}")
    print(f"{'━' * 50}")
    print(f"  Games run       : {len(result.game_results):,}")
    print(f"  Total P&L       : {result.total_pnl:>+12,.2f}")
    print(f"  Sharpe ratio    : {result.sharpe_ratio:>8.3f}")
    print(f"  Max drawdown    : {result.max_drawdown:>10.2f}")
    print(f"  Win rate        : {result.win_rate:>9.1%}")
    print(f"  Avg trades/game : {result.avg_trades_per_game:>8.1f}")
 
 
# ------------------------------------------------------------------
# Chart generation
# ------------------------------------------------------------------
 
def generate_all_charts(normal_result, adversarial_result):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
 
    os.makedirs("charts", exist_ok=True)
 
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "text.color":       "#e6edf3",
        "grid.color":       "#21262d",
        "grid.linestyle":   "--",
        "grid.alpha":       0.6,
        "lines.linewidth":  1.8,
        "font.family":      "monospace",
    })
 
    B, G, R, O, P = "#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff"
 
    # ---- Chart 1: Spread collapse + variance decay ----
    from src.game_env import CardGameState
    from src.ev_engine import compute_spread_width, compute_variance
 
    state = CardGameState()
    state.shuffle(seed=99)
    spreads, variances, x = [], [], []
 
    for step in range(N_STOP):
        d_left = N_STOP - step
        spreads.append(compute_spread_width(state, 0.5, n_to_draw=d_left))
        variances.append(compute_variance(state, n_to_draw=d_left))
        x.append(step)
        state.draw_card()
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spread Width vs Information Revealed", fontsize=14, color="#e6edf3", y=1.01)
 
    ax1.plot(x, spreads, color=B, lw=2)
    ax1.fill_between(x, spreads, alpha=0.15, color=B)
    ax1.set_xlabel("Cards Drawn")
    ax1.set_ylabel("Spread Width")
    ax1.set_title("Spread Collapses as Information Arrives", color="#8b949e", fontsize=10)
    ax1.axvline(x=30, color=O, linestyle=":", alpha=0.8, label="30 cards drawn")
    ax1.annotate(f"Start: {spreads[0]:.2f}", xy=(0, spreads[0]),
                 xytext=(1, spreads[0] + 0.05), color=O, fontsize=9)
    ax1.annotate(f"End: {spreads[-1]:.4f}", xy=(x[-1], spreads[-1]),
                 xytext=(x[-1] - 10, spreads[-1] + 0.2), color=O, fontsize=9)
    ax1.legend(framealpha=0.2)
    ax1.grid(True)
 
    ax2.plot(x, variances, color=P, lw=2)
    ax2.fill_between(x, variances, alpha=0.15, color=P)
    ax2.set_xlabel("Cards Drawn")
    ax2.set_ylabel("Variance of Settlement Sum")
    ax2.set_title(
        "Hypergeometric Variance Decay\n"
        "Var = n_settle × σ² × d / (R−1)",
        color="#8b949e", fontsize=10
    )
    ax2.grid(True)
 
    plt.tight_layout()
    plt.savefig("charts/1_spread_collapse.png", bbox_inches="tight", dpi=150)
    print("  Saved: charts/1_spread_collapse.png")
    plt.close()
 
    # ---- Chart 2: Demo game EV convergence ----
    demo_result = run_demo_game(k=0.5, n_stop=N_STOP, seed=7)
    snaps = demo_result.snapshots
 
    draws     = [s["cards_drawn"]         for s in snaps]
    evs       = [s["ev"]                  for s in snaps]
    bids      = [s["bid"]                 for s in snaps]
    asks      = [s["ask"]                 for s in snaps]
    true_sums = [s["true_remaining_sum"]  for s in snaps]
    spreads_d = [s["spread_width"]        for s in snaps]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Live Demo: EV Convergence & Spread Dynamics", fontsize=13, color="#e6edf3")
 
    ax1.fill_between(draws, bids, asks, alpha=0.2, color=B, label="Bid-Ask Band")
    ax1.plot(draws, evs, color=B, lw=2, label="EV (Mid-Price)")
    ax1.plot(draws, true_sums, color=G, lw=1.5, linestyle="--", label="True Settlement Sum")
    ax1.plot(draws, bids, color=B, lw=0.8, alpha=0.4)
    ax1.plot(draws, asks, color=B, lw=0.8, alpha=0.4)
    ax1.set_ylabel("Contract Value")
    ax1.set_title(
        "EV Converges to True Settlement Sum as Each Card Is Revealed",
        color="#8b949e", fontsize=10
    )
    ax1.legend(framealpha=0.2, loc="upper right")
    ax1.grid(True)
 
    ax2.bar(draws, spreads_d, color=P, alpha=0.75, width=0.8)
    ax2.set_xlabel("Cards Drawn")
    ax2.set_ylabel("Spread Width")
    ax2.set_title("Spread Narrows With Each Card Drawn", color="#8b949e", fontsize=10)
    ax2.grid(True)
 
    plt.tight_layout()
    plt.savefig("charts/2_ev_convergence.png", bbox_inches="tight", dpi=150)
    print("  Saved: charts/2_ev_convergence.png")
    plt.close()
 
    # ---- Chart 3: 10k game P&L curve ----
    pnls = np.array([r.pnl for r in normal_result.game_results])
    cumulative   = np.cumsum(pnls)
    n            = len(pnls)
    games        = np.arange(1, n + 1)
    running_max  = np.maximum.accumulate(cumulative)
    drawdowns    = running_max - cumulative
    window       = 300
    rolling_wins = np.convolve((pnls > 0).astype(float), np.ones(window) / window, mode="valid")
 
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.06)
 
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(games, cumulative, color=G, lw=1.5, label="Cumulative P&L")
    ax1.fill_between(games, cumulative, alpha=0.1, color=G)
    ax1.plot(games, running_max, color=B, lw=0.8, linestyle=":", alpha=0.6, label="Running Peak")
    mdd_end = int(np.argmax(drawdowns))
    ax1.axvspan(max(0, mdd_end - 400), mdd_end, alpha=0.1, color=R, label="Max DD Region")
    ax1.annotate(
        f"Sharpe:   {normal_result.sharpe_ratio:.2f}\n"
        f"Max DD:   {normal_result.max_drawdown:.1f}\n"
        f"Win Rate: {normal_result.win_rate:.1%}",
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9,
        color="#e6edf3", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#21262d", alpha=0.88, ec="#30363d")
    )
    ax1.set_title(
        f"P&L Across {n:,} Games  |  Total: {normal_result.total_pnl:,.0f}  |"
        f"  Avg {normal_result.avg_trades_per_game:.1f} trades/game",
        color="#8b949e", fontsize=10
    )
    ax1.set_ylabel("Cumulative P&L")
    ax1.legend(framealpha=0.2, fontsize=9)
    ax1.grid(True)
    ax1.set_xticklabels([])
 
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(games, -drawdowns, alpha=0.6, color=R)
    ax2.plot(games, -drawdowns, color=R, lw=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True)
    ax2.set_xticklabels([])
 
    ax3 = fig.add_subplot(gs[2])
    rolling_x = np.arange(window, n + 1)
    ax3.plot(rolling_x, rolling_wins, color=O, lw=1.2)
    ax3.axhline(y=0.5, color="#444", linestyle="--", alpha=0.8)
    ax3.set_ylabel(f"Win Rate\n({window}g rolling)")
    ax3.set_xlabel("Game Number")
    ax3.set_ylim(0.2, 0.95)
    ax3.grid(True)
 
    plt.savefig("charts/3_pnl_curve.png", bbox_inches="tight", dpi=150)
    print("  Saved: charts/3_pnl_curve.png")
    plt.close()
 
    # ---- Chart 4: Adversarial comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Market Maker Performance: Normal vs Adversarial Informed Flow",
        fontsize=13, color="#e6edf3"
    )
    for ax, res, label, color in zip(
        axes,
        [normal_result, adversarial_result],
        ["Normal Flow (60% Noise / 30% Soft-Informed / 10% Hard-Informed)",
         "Adversarial (80% Hard-Informed Traders)"],
        [G, R],
    ):
        ps  = np.array([r.pnl for r in res.game_results])
        cum = np.cumsum(ps)
        gs_ = np.arange(1, len(ps) + 1)
        ax.plot(gs_, cum, color=color, lw=1.5)
        ax.fill_between(gs_, cum, alpha=0.12, color=color)
        ax.annotate(
            f"Sharpe:   {res.sharpe_ratio:.2f}\n"
            f"Win Rate: {res.win_rate:.1%}\n"
            f"Total P&L: {res.total_pnl:,.0f}",
            xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9,
            color="#e6edf3", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#21262d", alpha=0.85, ec="#30363d")
        )
        ax.set_title(label, color="#8b949e", fontsize=9)
        ax.set_xlabel("Game Number")
        ax.set_ylabel("Cumulative P&L")
        ax.grid(True)
 
    plt.tight_layout()
    plt.savefig("charts/4_adversarial.png", bbox_inches="tight", dpi=150)
    print("  Saved: charts/4_adversarial.png")
    plt.close()
 
 
# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="EV Market Maker Simulation")
    parser.add_argument("--demo",      action="store_true", help="Narrated demo only")
    parser.add_argument("--sim",       action="store_true", help="Simulation + charts only")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart rendering")
    parser.add_argument("--n-games",   type=int, default=10_000, help="Games to simulate")
    args = parser.parse_args()
 
    run_all = not args.demo and not args.sim
 
    # ---- Live demo ----
    if args.demo or run_all:
        run_narrated_demo(show_steps=20)
 
    # ---- Simulation ----
    if args.sim or run_all:
        print(f"\n  Running {args.n_games:,} game simulation (normal flow)...")
        normal = run_simulation(
            n_games=args.n_games, k=0.5, n_stop=N_STOP,
            noise_weight=0.6, soft_weight=0.3, hard_weight=0.1,
            seed=42, verbose=True,
        )
        print_summary(normal, "Normal Flow Simulation")
 
        print("\n  Running adversarial stress test (1,000 games)...")
        adversarial = run_adversarial_test(n_games=1000, k=0.5)
        print_summary(adversarial, "Adversarial Stress Test")
 
        if not args.no_charts:
            print("\n  Generating charts...")
            generate_all_charts(normal, adversarial)
            print("  All charts saved to charts/")
 
 
if __name__ == "__main__":
    main()
