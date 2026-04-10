
import pytest
import numpy as np
 
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
 
from src.game_env import CardGameState, DiceGameState
from src.ev_engine import (
    compute_ev,
    compute_variance,
    compute_spread_width,
    compute_quote,
    compute_dice_ev,
    compute_dice_variance,
)
 
 
# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
 
@pytest.fixture
def fresh_state():
    state = CardGameState()
    state.shuffle(seed=42)
    return state
 
 
@pytest.fixture
def mid_game_state():
    """Draw 20 cards and return the state."""
    state = CardGameState()
    state.shuffle(seed=42)
    for _ in range(20):
        state.draw_card()
    return state
 
 
@pytest.fixture
def near_end_state():
    """Draw 51 cards — only one left."""
    state = CardGameState()
    state.shuffle(seed=42)
    for _ in range(51):
        state.draw_card()
    return state
 
 
# ------------------------------------------------------------------
# EV tests
# ------------------------------------------------------------------
 
class TestComputeEV:
 
    def test_fresh_deck_ev(self, fresh_state):
        """
        Full 52-card deck: 4 of each value 1-9, and 16 tens (10,J,Q,K).
        Sum = 4*(1+2+...+9) + 16*10 = 4*45 + 160 = 340
        EV of remaining sum at start = 340.
        """
        ev = compute_ev(fresh_state)
        assert abs(ev - 340.0) < 1e-6, f"Expected EV=340, got {ev}"
 
    def test_ev_decreases_as_cards_drawn(self, fresh_state):
        """After drawing cards, EV of remaining sum should decrease (on average)."""
        ev_before = compute_ev(fresh_state)
        for _ in range(10):
            fresh_state.draw_card()
        ev_after = compute_ev(fresh_state)
        # EV should be less since fewer cards remain
        assert ev_after < ev_before
 
    def test_ev_at_empty_deck(self):
        state = CardGameState()
        state.shuffle()
        for _ in range(52):
            state.draw_card()
        assert compute_ev(state) == 0.0
 
    def test_ev_is_non_negative(self, mid_game_state):
        ev = compute_ev(mid_game_state)
        assert ev >= 0.0
 
 
# ------------------------------------------------------------------
# Variance tests
# ------------------------------------------------------------------
 
class TestComputeVariance:
 
    def test_variance_at_start_is_positive(self, fresh_state):
        var = compute_variance(fresh_state)
        assert var > 0.0
 
    def test_variance_collapses_near_end(self, near_end_state):
        """One card left — variance should be 0 or very close to 0."""
        var = compute_variance(near_end_state)
        assert var == 0.0, f"Expected variance=0 with one card left, got {var}"
 
    def test_variance_decreases_overall(self, fresh_state):
        """
        On average, drawing cards reduces variance.
        We draw 30 cards and check that variance at the end < start.
        """
        var_start = compute_variance(fresh_state)
        for _ in range(30):
            fresh_state.draw_card()
        var_end = compute_variance(fresh_state)
        assert var_end < var_start
 
    def test_variance_is_non_negative(self, mid_game_state):
        var = compute_variance(mid_game_state)
        assert var >= 0.0
 
    def test_variance_at_two_cards_left(self):
        """Two cards left: known values, easy to verify by hand."""
        state = CardGameState()
        state.shuffle(seed=0)
        # draw until 2 remain
        while state.cards_remaining() > 2:
            state.draw_card()
 
        remaining = state.remaining
        assert len(remaining) == 2
 
        # manual calculation
        n = 2
        N = 52
        pop_var = np.var(remaining)
        fpc = (N - n) / (N - 1)
        expected_var = n * pop_var * fpc
 
        computed_var = compute_variance(state)
        assert abs(computed_var - expected_var) < 1e-10
 
 
# ------------------------------------------------------------------
# Spread / quote tests
# ------------------------------------------------------------------
 
class TestQuoting:
 
    def test_spread_centered_on_ev(self, fresh_state):
        """Bid and ask should be symmetrically around EV."""
        ev = compute_ev(fresh_state)
        bid, ask = compute_quote(fresh_state, k=0.5)
        mid = (bid + ask) / 2.0
        assert abs(mid - ev) < 1e-6
 
    def test_bid_less_than_ask(self, fresh_state):
        bid, ask = compute_quote(fresh_state, k=0.5)
        assert bid < ask
 
    def test_spread_width_shrinks_as_cards_drawn(self, fresh_state):
        spread_start = compute_spread_width(fresh_state, k=0.5)
        for _ in range(40):
            fresh_state.draw_card()
        spread_end = compute_spread_width(fresh_state, k=0.5)
        assert spread_end < spread_start
 
    def test_spread_is_zero_at_end(self, near_end_state):
        spread = compute_spread_width(near_end_state, k=0.5)
        assert spread == 0.0
 
    def test_wider_k_means_wider_spread(self, mid_game_state):
        bid_narrow, ask_narrow = compute_quote(mid_game_state, k=0.3)
        bid_wide, ask_wide = compute_quote(mid_game_state, k=1.0)
        spread_narrow = ask_narrow - bid_narrow
        spread_wide = ask_wide - bid_wide
        assert spread_wide > spread_narrow
 
 
# ------------------------------------------------------------------
# Dice game tests
# ------------------------------------------------------------------
 
class TestDiceEV:
 
    def test_dice_ev_at_start(self):
        state = DiceGameState(n_dice=5, sides=6)
        ev = compute_dice_ev(state)
        expected = 5 * 3.5
        assert abs(ev - expected) < 1e-9
 
    def test_dice_ev_after_rolls(self):
        state = DiceGameState(n_dice=5, sides=6)
        state.roll()
        state.roll()
        ev = compute_dice_ev(state)
        expected = 3 * 3.5  # 3 dice remaining
        assert abs(ev - expected) < 1e-9
 
    def test_dice_variance_at_start(self):
        state = DiceGameState(n_dice=5, sides=6)
        var = compute_dice_variance(state)
        expected = 5 * (6**2 - 1) / 12
        assert abs(var - expected) < 1e-9
 
    def test_dice_variance_decreases_with_rolls(self):
        state = DiceGameState(n_dice=5, sides=6)
        var_start = compute_dice_variance(state)
        state.roll()
        state.roll()
        var_mid = compute_dice_variance(state)
        assert var_mid < var_start
 
 
# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------
 
class TestEdgeCases:
 
    def test_full_deck_mean_is_correct(self):
        """52-card deck: mean card value = 340/52 ≈ 6.538"""
        state = CardGameState()
        mean = np.mean(state.deck)
        expected = 340 / 52
        assert abs(mean - expected) < 1e-9
 
    def test_deck_has_52_cards(self):
        state = CardGameState()
        assert len(state.deck) == 52
 
    def test_deck_has_correct_value_counts(self):
        """Each value 1-9 appears 4x, value 10 appears 16x (10+J+Q+K)."""
        from collections import Counter
        state = CardGameState()
        counts = Counter(state.deck)
        for v in range(1, 10):
            assert counts[v] == 4, f"Value {v} should appear 4 times, got {counts[v]}"
        assert counts[10] == 16, f"Value 10 should appear 16 times, got {counts[10]}"
 
    def test_reset_restores_full_deck(self):
        state = CardGameState()
        for _ in range(30):
            state.draw_card()
        state.reset()
        assert state.cards_remaining() == 52
        assert state.cards_drawn() == 0
 