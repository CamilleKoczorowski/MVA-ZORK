"""
Student MCP Server for Text Adventure Games

This is your MCP server submission. Implement the tools that your agent
will use to play text adventure games.

Required tool:
    play_action(action: str) -> str
        Execute a game command and return the result.

Recommended tools:
    memory() -> str
        Return current game state, score, and recent history.
    
    inventory() -> str  
        Return the player's current inventory.
    
    get_map() -> str
        Return a map of explored locations.

Test your server with:
    fastmcp dev submission_template/mcp_server.py

Then open the MCP Inspector in your browser to test the tools interactively.
"""

import sys
import os

# Add parent directory to path to import games module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv


# =============================================================================
# Create the MCP Server
# =============================================================================

mcp = FastMCP("Student Text Adventure Server")


# =============================================================================
# Game State Management
# =============================================================================

class GameManager:
    """
    Manages the text adventure game state.
    
    TODO: Extend this class to track:
    - Action history (for memory tool)
    - Explored locations (for mapping)
    - Current score and moves
    """
    
    def __init__(self):
        self.env: TextAdventureEnv = None
        self.state = None
        self.game_name: str = ""
        # TODO: Add more state tracking
        # self.history: list[tuple[str, str]] = []
        # self.explored_locations: dict[str, set[str]] = {}
        # self.current_location: str = ""
    
    def initialize(self, game: str = "zork1"):
        """Initialize or reset the game."""
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()
        # TODO: Reset your state tracking here
        return self.state.observation
    
    def step(self, action: str) -> str:
        """Execute an action and return the result."""
        if self.env is None:
            self.initialize()
        
        self.state = self.env.step(action)
        
        # TODO: Update your state tracking here
        # self.history.append((action, self.state.observation))
        # Update location tracking, etc.
        
        return self.state.observation
    
    def get_score(self) -> int:
        """Get current score."""
        return self.state.score if self.state else 0
    
    def get_moves(self) -> int:
        """Get number of moves taken."""
        return self.state.moves if self.state else 0


# Global game manager
_game = GameManager()


def get_game() -> GameManager:
    """Get or initialize the game manager."""
    global _game
    if _game.env is None:
        # Get game from environment variable (set by evaluator)
        game = os.environ.get("GAME", "zork1")
        _game.initialize(game)
    return _game


# =============================================================================
# MCP Tools - IMPLEMENT THESE
# =============================================================================

@mcp.tool()
def play_action(action: str) -> str:
    """
    Execute a game command and return the result.
    
    This is the main tool for interacting with the game.
    
    Args:
        action: The command to execute (e.g., "north", "take lamp", "open mailbox")
        
    Returns:
        The game's response to the action
        
    Valid commands include:
        - Movement: north, south, east, west, up, down, enter, exit
        - Objects: take <item>, drop <item>, open <thing>, examine <thing>
        - Other: look, inventory, read <thing>, turn on lamp
    """
    game = get_game()
    
    # TODO: You might want to add action validation here
    # TODO: You might want to include score changes in the response
    
    result = game.step(action)
    
    # Optional: Append score info
    # result += f"\n[Score: {game.get_score()} | Moves: {game.get_moves()}]"
    
    return result


# TODO: Implement additional tools to help your agent

# @mcp.tool()
# def memory() -> str:
#     """
#     Get the current game state summary.
#     
#     Returns:
#         A summary including current location, score, moves, and recent history
#     """
#     game = get_game()
#     # TODO: Return useful state information
#     pass


# @mcp.tool()
# def inventory() -> str:
#     """
#     Check what the player is carrying.
#     
#     Returns:
#         List of items in the player's inventory
#     """
#     game = get_game()
#     result = game.step("inventory")
#     return result


# @mcp.tool()
# def get_map() -> str:
#     """
#     Get a map of explored locations.
#     
#     Returns:
#         A text representation of explored locations and connections
#     """
#     game = get_game()
#     # TODO: Return map of explored locations
#     pass


# @mcp.tool()
# def get_valid_actions() -> str:
#     """
#     Get a list of likely valid actions from the current location.
#     
#     Returns:
#         List of actions that might work here
#     """
#     # This is a hint: Jericho provides get_valid_actions()
#     game = get_game()
#     if game.env and game.env.env:
#         valid = game.env.env.get_valid_actions()
#         return "Valid actions: " + ", ".join(valid[:20])
#     return "Could not determine valid actions"


# =============================================================================
# Run the server
# =============================================================================

if __name__ == "__main__":
    # This runs the server with stdio transport (for MCP clients)
    mcp.run()
