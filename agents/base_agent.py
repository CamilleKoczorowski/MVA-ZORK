"""
Base Agent Abstract Class

Defines the interface that all text adventure agents must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from games.zork_env import GameState


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str = "BaseAgent"
    max_history: int = 20  # Maximum number of past interactions to remember
    verbose: bool = False


class BaseAgent(ABC):
    """
    Abstract base class for text adventure agents.
    
    Students should extend this class and implement the `choose_action` method.
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.history: list[tuple[str, str, GameState]] = []  # (action, observation, state)
    
    @abstractmethod
    def choose_action(self, observation: str, game_state: GameState) -> str:
        """
        Choose the next action based on the current observation and game state.
        
        Args:
            observation: The text observation from the game
            game_state: The current GameState object with score, inventory, etc.
            
        Returns:
            A string action to take in the game (e.g., "go north", "take lamp")
        """
        pass
    
    def update_history(self, action: str, observation: str, game_state: GameState):
        """
        Update the agent's history after taking an action.
        
        Args:
            action: The action that was taken
            observation: The resulting observation
            game_state: The resulting game state
        """
        self.history.append((action, observation, game_state))
        
        # Keep history bounded
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]
    
    def reset(self):
        """Reset the agent's internal state for a new game."""
        self.history = []
    
    def get_history_text(self) -> str:
        """Get a text summary of recent history for context."""
        if not self.history:
            return "No previous actions taken."
        
        lines = []
        for action, observation, state in self.history[-10:]:  # Last 10 actions
            lines.append(f"> {action}")
            # Truncate long observations
            obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
            lines.append(obs_preview)
            lines.append(f"[Score: {state.score}, Moves: {state.moves}]")
            lines.append("")
        
        return "\n".join(lines)
