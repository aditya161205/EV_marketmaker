"""
game_env.py
 
Defines the game environments the market maker operates on.
Currently supports a standard 52-card deck and a dice game.
 
Card values:
    Ace = 1, 2-10 = face value, Jack/Queen/King = 10
"""
 
import random
from dataclasses import dataclass, field
from typing import List, Optional
 
 
# ------------------------------------------------------------------
# Card deck environment
# ------------------------------------------------------------------
 
def build_deck() -> List[int]:
    """
    Returns a standard 52-card deck as a list of integer values.
    Suits don't matter — only the numeric value of each card.
    """
    values = []
 
    for rank in range(1, 14):  # 1=Ace, 11=Jack, 12=Queen, 13=King
        card_val = min(rank, 10)  # face cards capped at 10
        values.extend([card_val] * 4)  # four suits
 
    return values
 
 
@dataclass
class CardGameState:

    deck: List[int] = field(default_factory=build_deck)
    drawn: List[int] = field(default_factory=list)
    remaining: List[int] = field(default_factory=list)
 
    def __post_init__(self):
        # remaining starts as a copy of the full deck
        self.remaining = list(self.deck)
 
    def shuffle(self, seed: Optional[int] = None):
        """Shuffle the deck to start a new game."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.remaining)
        self.drawn = []
 
    def draw_card(self) -> Optional[int]:
        """
        Draw the next card from the deck.
        Returns None if the deck is empty.
        """
        if not self.remaining:
            return None
 
        card = self.remaining.pop(0)
        self.drawn.append(card)
        return card
 
    def cards_remaining(self) -> int:
        return len(self.remaining)
 
    def cards_drawn(self) -> int:
        return len(self.drawn)
 
    def is_finished(self) -> bool:
        return len(self.remaining) == 0
 
    def true_remaining_sum(self) -> int:
        """The actual sum of cards still in the deck. Used for P&L calculation."""
        return sum(self.remaining)
 
    def reset(self):
        """Reset to a fresh shuffled game."""
        self.remaining = list(self.deck)
        self.drawn = []
        self.shuffle()
 
    def __repr__(self):
        return (
            f"CardGameState(drawn={self.cards_drawn()}, "
            f"remaining={self.cards_remaining()}, "
            f"last_card={self.drawn[-1] if self.drawn else None})"
        )
 
 
# ------------------------------------------------------------------
# Dice game environment (bonus environment for extension)
# ------------------------------------------------------------------
 
@dataclass
class DiceGameState:

    n_dice: int = 5
    rolls: List[int] = field(default_factory=list)
    sides: int = 6
 
    @property
    def die_ev(self) -> float:
        return (self.sides + 1) / 2.0
 
    @property
    def die_var(self) -> float:
        # Var[uniform discrete {1,...,n}] = (n^2 - 1) / 12
        return (self.sides ** 2 - 1) / 12.0
 
    def roll(self) -> Optional[int]:
        if len(self.rolls) >= self.n_dice:
            return None
        result = random.randint(1, self.sides)
        self.rolls.append(result)
        return result
 
    def dice_remaining(self) -> int:
        return self.n_dice - len(self.rolls)
 
    def is_finished(self) -> bool:
        return len(self.rolls) >= self.n_dice
 
    def true_remaining_sum(self) -> int:
        """We don't know future dice, so this isn't available mid-game."""
        raise NotImplementedError("Future dice rolls are not observable")
 
    def reset(self):
        self.rolls = []
 