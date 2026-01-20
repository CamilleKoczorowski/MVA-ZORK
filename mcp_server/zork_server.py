"""
Text Adventure MCP Server - Exposes text adventure games via Model Context Protocol.

This server allows any MCP-compatible agent to play Zork and other text adventure 
games using tools for game actions, memory, mapping, and inventory.

Uses FastMCP for simple, Pythonic MCP server implementation.

Usage:
    # Run directly (stdio transport) - default game is zork1
    python mcp_server/zork_server.py
    
    # Run with a different game
    GAME=zork2 python mcp_server/zork_server.py
    GAME=advent python mcp_server/zork_server.py
    GAME=enchanter python mcp_server/zork_server.py
    
    # Use with FastMCP dev tools
    fastmcp dev mcp_server/zork_server.py
    
    # Connect from an MCP client
    from fastmcp import Client
    async with Client("mcp_server/zork_server.py") as client:
        result = await client.call_tool("play_action", {"action": "look"})
"""

import sys
import os

# Add parent directory to path to import games module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv, list_available_games


# Get game from environment variable (default: zork1)
INITIAL_GAME = os.environ.get("GAME", "zork1")

# Create the MCP server
mcp = FastMCP("Text Adventure Server")


class GameState:
    """Manages the text adventure game state and exploration data."""
    
    def __init__(self, game: str = "zork1"):
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()
        self.history: list[tuple[str, str]] = []
        self.explored_locations: dict[str, set[str]] = {}  # location -> set of exits
        self.current_location: str = self._extract_location(self.state.observation)
    
    def _extract_location(self, observation: str) -> str:
        """Extract location name from observation (usually first line)."""
        lines = observation.strip().split('\n')
        return lines[0] if lines else "Unknown"
    
    def take_action(self, action: str) -> str:
        """Execute a game action and return the result."""
        self.state = self.env.step(action)
        result = self.state.observation
        
        # Track history
        self.history.append((action, result))
        if len(self.history) > 50:
            self.history = self.history[-50:]
        
        # Update map
        new_location = self._extract_location(result)
        if action in ["north", "south", "east", "west", "up", "down", 
                      "enter", "exit", "n", "s", "e", "w", "u", "d"]:
            if self.current_location not in self.explored_locations:
                self.explored_locations[self.current_location] = set()
            if new_location != self.current_location:
                self.explored_locations[self.current_location].add(f"{action} -> {new_location}")
        self.current_location = new_location
        
        return result
    
    def get_memory(self) -> str:
        """Get a summary of current game state."""
        recent = self.history[-5:] if self.history else []
        recent_str = "\n".join([f"  > {a} → {r[:60]}..." for a, r in recent]) if recent else "  (none yet)"
        
        return f"""Current State:
- Location: {self.current_location}
- Score: {self.state.score} points
- Moves: {self.state.moves}
- Game: {self.game_name}

Recent Actions:
{recent_str}

Current Observation:
{self.state.observation}"""
    
    def get_map(self) -> str:
        """Get a map of explored locations."""
        if not self.explored_locations:
            return "Map: No locations explored yet. Try moving around!"
        
        lines = ["Explored Locations and Exits:"]
        for loc, exits in sorted(self.explored_locations.items()):
            lines.append(f"\n* {loc}")
            for exit_info in sorted(exits):
                lines.append(f"    -> {exit_info}")
        
        lines.append(f"\n[Current] {self.current_location}")
        return "\n".join(lines)
    
    def get_inventory(self) -> str:
        """Get current inventory."""
        items = self.state.inventory if hasattr(self.state, 'inventory') and self.state.inventory else []
        
        if not items:
            return "Inventory: You are empty-handed."
        
        item_names = []
        for item in items:
            item_str = str(item)
            # Handle Jericho's object format: "leaflet Parent4 Sibling0..."
            # Look for "Parent" (case-insensitive) to find where metadata starts
            item_lower = item_str.lower()
            if "parent" in item_lower:
                idx = item_lower.index("parent")
                name = item_str[:idx].strip()
                # Remove leading "obj123: " if present
                if ":" in name:
                    name = name.split(":", 1)[1].strip()
                item_names.append(name)
            elif ":" in item_str:
                name = item_str.split(":")[1].strip()
                item_names.append(name)
            else:
                item_names.append(item_str)
        
        return f"Inventory: {', '.join(item_names)}"
    
    def get_valid_actions(self) -> str:
        """Get list of valid actions in current state."""
        try:
            valid = self.env.get_valid_actions() if hasattr(self.env, 'get_valid_actions') else []
            if valid:
                return f"Valid actions: {', '.join(valid[:20])}"
        except Exception:
            pass
        return "Valid actions: Try standard commands like look, north, south, east, west, take <item>, open <thing>"


# Global game state (initialized on first use)
_game_state: GameState | None = None


def get_game() -> GameState:
    """Get or initialize the game state."""
    global _game_state
    if _game_state is None:
        _game_state = GameState(INITIAL_GAME)
    return _game_state


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
def play_action(action: str) -> str:
    """
    Execute a game action in the text adventure.
    
    Common commands:
    - Movement: north, south, east, west, up, down, enter, exit (or n, s, e, w, u, d)
    - Objects: take <item>, drop <item>, open <thing>, close <thing>, put <item> in <container>
    - Look: look, examine <thing>, read <thing>
    - Combat: attack <enemy> with <weapon>
    - Light: turn on lamp, light match
    - Other: wait, score, inventory
    
    Args:
        action: The command to execute (e.g., 'north', 'take lamp', 'open mailbox')
    
    Returns:
        The game's response to your action
    """
    game = get_game()
    result = game.take_action(action)
    
    # Add score info if points were earned
    score_info = ""
    if game.state.reward > 0:
        score_info = f"\n\n+{game.state.reward} points! (Total: {game.state.score})"
    
    done_info = ""
    if game.state.done:
        done_info = "\n\nGAME OVER"
    
    return result + score_info + done_info


@mcp.tool()
def memory() -> str:
    """
    Get a summary of the current game state.
    
    Returns your location, score, moves, recent actions, and current observation.
    Use this to understand where you are and what happened recently.
    Very useful for avoiding loops and tracking progress.
    """
    return get_game().get_memory()


@mcp.tool()
def get_map() -> str:
    """
    Get a map showing all locations you have explored and the connections between them.
    
    Useful for navigation and planning routes back to previous locations.
    The map builds up as you explore more of the game world.
    """
    return get_game().get_map()


@mcp.tool()
def inventory() -> str:
    """
    Check what items you are currently carrying.
    
    Essential before trying to use, drop, or interact with items.
    Most games have an inventory limit, so manage your items wisely.
    """
    return get_game().get_inventory()


@mcp.tool()
def valid_actions() -> str:
    """
    Get a list of valid actions available in the current game state.
    
    Helpful when stuck or unsure what commands the game accepts.
    Note: This may not include all possible actions, just common ones.
    """
    return get_game().get_valid_actions()


@mcp.tool()
def reset_game(game: str = "zork1") -> str:
    """
    Reset the game to the beginning or switch to a different game.
    
    Use this to start over if you get stuck, die, or want to try a different game.
    
    Args:
        game: Game name (e.g., 'zork1', 'zork2', 'advent', 'enchanter')
              Use list_games() to see available options.
    
    Returns:
        The initial game text
    """
    global _game_state
    try:
        _game_state = GameState(game)
        return f"Game reset to {game}.\n\n{_game_state.state.observation}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def list_games() -> str:
    """
    List all available text adventure games.
    
    Returns:
        List of game names that can be passed to reset_game()
    """
    games = list_available_games()
    return f"Available games ({len(games)} total):\n" + ", ".join(games)


@mcp.tool()
def hint() -> str:
    """
    Get a hint about what to do next based on your current situation.
    
    Provides general guidance without spoiling puzzle solutions.
    """
    game = get_game()
    location = game.current_location.lower()
    inv = game.get_inventory().lower()
    observation = game.state.observation.lower()
    
    hints = []
    
    # Darkness detection (common in many games)
    if "dark" in location or "dark" in observation or "pitch black" in observation:
        hints.append("It's dangerous in the dark! You need a light source.")
        hints.append("If you have a lamp, try 'turn on lamp'.")
    
    # Common items to look for
    if "lamp" in observation and "lamp" not in inv:
        hints.append("There's a lamp here - light sources are essential!")
    if "lantern" in observation and "lantern" not in inv:
        hints.append("There's a lantern here - you'll need light for dark areas!")
    if "sword" in observation and "sword" not in inv:
        hints.append("A sword might be useful for combat encounters.")
    if "key" in observation and "key" not in inv:
        hints.append("A key might unlock something important.")
    
    # Container hints
    if any(word in observation for word in ["mailbox", "chest", "box", "container", "cabinet"]):
        hints.append("Try opening containers to find hidden items.")
    
    # Door/window hints
    if "door" in observation or "window" in observation:
        hints.append("There might be a way in or out here. Try 'open' commands.")
    
    # General hints if nothing specific found
    if not hints:
        hints.append("Explore all directions: north, south, east, west, up, down.")
        hints.append("Examine interesting objects with 'examine <thing>'.")
        hints.append("Pick up useful items with 'take <item>'.")
        hints.append("Open containers and read documents for clues.")
    
    return "Hints:\\n" + "\\n".join(f"  - {h}" for h in hints)


# ============================================================================
# MCP Resources
# ============================================================================

@mcp.resource("game://state")
def get_state_resource() -> str:
    """Current game state as a resource."""
    return get_game().get_memory()


@mcp.resource("game://history")
def get_history_resource() -> str:
    """Complete action history as a resource."""
    game = get_game()
    if not game.history:
        return "No actions taken yet."
    lines = [f"{i+1}. {action} -> {result[:80]}..." for i, (action, result) in enumerate(game.history)]
    return "\n".join(lines)


@mcp.resource("game://map")
def get_map_resource() -> str:
    """Explored map as a resource."""
    return get_game().get_map()


# ============================================================================
# Game Prompt (for agents)
# ============================================================================

GAME_PROMPT = """You are playing a classic text adventure game.

## YOUR GOAL
Explore the world, solve puzzles, collect treasures, and maximize your score.

## VALID COMMANDS (use ONLY these exact verbs)

Movement:
  north, south, east, west, up, down (or n, s, e, w, u, d)
  enter, exit, climb, cross, go <direction>

Looking:
  look, examine <thing>, look at <thing>, look in <thing>, read <thing>

Objects:
  take <item>, drop <item>, pick up <item>
  open <thing>, close <thing>, unlock <thing> with <key>
  put <item> in <container>, give <item> to <person>

Light:
  turn on lamp, turn off lamp, light match

Combat:
  attack <enemy> with <weapon>, kill <enemy> with <weapon>

Other:
  inventory (or i), wait (or z), score
  push <thing>, pull <thing>, move <thing>
  tie <rope> to <thing>, eat <food>, wave <item>

## FORBIDDEN VERBS (these will NOT work):
  check, inspect, search, investigate, grab, pick, use, interact,
  go to, walk to, head to, travel, proceed

## STRATEGY TIPS
1. Explore systematically - check all directions
2. Read everything - open containers, read documents, examine objects
3. Use get_map() to track explored locations
4. Light is essential - find a light source before dark areas!
5. Manage inventory - you can only carry limited items

## GETTING STARTED
1. Call memory() to see your current state
2. Explore your starting area thoroughly
3. Pick up useful items (light sources, weapons, keys)

Good luck!
"""


def get_game_prompt(game: str = "zork1") -> str:
    """Get the system prompt for playing text adventures."""
    prompt = GAME_PROMPT
    prompt += f"\n\nNote: Currently playing {game}. Use list_games() to see all 57 available games."
    return prompt


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    mcp.run()
