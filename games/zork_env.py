"""
Text Adventure Game Environment

Provides a clean interface to text adventure games via Jericho.
Supports Zork and many other classic Z-machine games.
"""

from jericho import FrotzEnv
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os


@dataclass
class GameState:
    """Represents the current state of the game."""
    observation: str
    score: int
    max_score: int
    moves: int
    done: bool
    reward: int  # Points gained from last action
    inventory: list[str]
    location: str


def get_default_games_dir() -> Path:
    """Get the default directory containing game files."""
    project_root = Path(__file__).parent.parent
    return project_root / "z-machine-games-master" / "jericho-game-suite"


def discover_games(games_dir: Optional[Path] = None) -> dict[str, Path]:
    """
    Discover all available Z-machine games in the games directory.
    
    Args:
        games_dir: Directory to search for games (default: jericho-game-suite)
        
    Returns:
        Dictionary mapping game name (without extension) to full path
    """
    if games_dir is None:
        games_dir = get_default_games_dir()
    
    games_dir = Path(games_dir)
    if not games_dir.exists():
        return {}
    
    games = {}
    # Find all Z-machine game files (.z3, .z4, .z5, .z8)
    for ext in ["*.z3", "*.z4", "*.z5", "*.z8"]:
        for game_path in games_dir.glob(ext):
            # Use stem (filename without extension) as game name
            game_name = game_path.stem.lower()
            games[game_name] = game_path
    
    return dict(sorted(games.items()))


def list_available_games(games_dir: Optional[Path] = None) -> list[str]:
    """Return a sorted list of available game names."""
    return list(discover_games(games_dir).keys())


class TextAdventureEnv:
    """Wrapper around Jericho's FrotzEnv for text adventure games."""
    
    def __init__(self, game: str = "zork1", games_dir: Optional[str] = None):
        """
        Initialize the text adventure environment.
        
        Args:
            game: Game name (e.g., 'zork1', 'advent', 'enchanter')
                  Can also be a full path to a .z* file
            games_dir: Directory containing game files (optional)
        """
        # Check if game is a full path
        if os.path.isfile(game):
            game_path = Path(game)
            self.game = game_path.stem
        else:
            # Look up game by name
            games_path = Path(games_dir) if games_dir else None
            available_games = discover_games(games_path)
            
            if game.lower() not in available_games:
                available = list(available_games.keys())[:20]
                raise ValueError(
                    f"Unknown game: {game}. "
                    f"Available: {', '.join(available)}... "
                    f"({len(available_games)} total)"
                )
            
            game_path = available_games[game.lower()]
            self.game = game.lower()
        
        self.env = FrotzEnv(str(game_path))
        self.game_path = game_path
        self._last_score = 0
        self._history: list[tuple[str, str]] = []  # (action, observation) pairs
    
    def reset(self) -> GameState:
        """Reset the game to the beginning."""
        observation, info = self.env.reset()
        self._last_score = 0
        self._history = []
        return self._make_game_state(observation, info, done=False, reward=0)
    
    def step(self, action: str) -> GameState:
        """
        Take an action in the game.
        
        Args:
            action: The text command to execute (e.g., "go north", "take lamp")
            
        Returns:
            GameState with the result of the action
        """
        observation, reward, done, info = self.env.step(action)
        
        # Track reward as score change
        current_score = info.get('score', 0)
        reward = current_score - self._last_score
        self._last_score = current_score
        
        # Record history
        self._history.append((action, observation))
        
        return self._make_game_state(observation, info, done, reward)
    
    def _make_game_state(self, observation: str, info: dict, done: bool, reward: int) -> GameState:
        """Create a GameState from the environment info."""
        # Try to get inventory and location (may fail without spacy)
        try:
            inventory = [str(obj) for obj in self.env.get_inventory()]
        except Exception:
            inventory = []
        
        try:
            location = str(self.env.get_player_location())
        except Exception:
            location = "Unknown"
        
        return GameState(
            observation=observation,
            score=info.get('score', 0),
            max_score=self.env.get_max_score(),
            moves=info.get('moves', 0),
            done=done,
            reward=reward,
            inventory=inventory,
            location=location,
        )
    
    def get_history(self) -> list[tuple[str, str]]:
        """Get the history of (action, observation) pairs."""
        return self._history.copy()
    
    def get_valid_actions(self) -> list[str]:
        """
        Get a list of valid actions for the current state.
        Note: This requires spacy to be properly installed.
        """
        try:
            return self.env.get_valid_actions()
        except Exception:
            # Return common actions if spacy isn't available
            return [
                "north", "south", "east", "west",
                "up", "down", "look", "inventory",
                "take all", "open mailbox", "read"
            ]
    
    def save_state(self):
        """Save the current game state."""
        return self.env.get_state()
    
    def load_state(self, state):
        """Load a previously saved game state."""
        self.env.set_state(state)
    
    def get_walkthrough(self) -> list[str]:
        """Get the walkthrough for the game (for debugging/comparison only)."""
        return self.env.get_walkthrough()


# Alias for backwards compatibility
ZorkEnvironment = TextAdventureEnv


# Example usage
if __name__ == "__main__":
    import sys
    
    # List available games
    games = list_available_games()
    print(f"Available games ({len(games)} total):")
    print(f"  {', '.join(games[:15])}...")
    print()
    
    # Use command line arg or default to zork1
    game = sys.argv[1] if len(sys.argv) > 1 else "zork1"
    
    env = TextAdventureEnv(game)
    state = env.reset()
    
    print(f"=== {env.game.upper()} ===")
    print(f"Max Score: {state.max_score}")
    print(f"\n{state.observation}")
    print(f"\nValid actions: {env.get_valid_actions()[:10]}...")
    
    # Try a few actions
    for action in ["look", "inventory"]:
        print(f"\n> {action}")
        state = env.step(action)
        print(state.observation)
        print(f"Score: {state.score}, Reward: {state.reward}")
