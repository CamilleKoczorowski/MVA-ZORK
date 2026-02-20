"""
Student MCP Server for Text Adventure Games - Improved Implementation

Improvements over example_submission/mcp_server.py:
1. memory() returns failed actions per location (key for agent anti-loop)
2. get_location() tool for clean location name retrieval
3. Action deduplication tracking exposed to agent
4. Score change detection in play_action response
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv

INITIAL_GAME = os.environ.get("GAME", "lostpig")

mcp = FastMCP("Text Adventure Server - Improved")


# =============================================================================
# Game State
# =============================================================================

class GameState:
    """
    Extended game state with per-location failed action tracking.
    """

    def __init__(self, game: str = "lostpig"):
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()

        # Full action/observation history (capped at 100)
        self.history: list[tuple[str, str]] = []
        # Per-location: set of actions with no effect
        self.failed_actions: dict[str, set[str]] = defaultdict(set)
        # Per-location: known exits {direction: destination}
        self.location_graph: dict[str, dict[str, str]] = defaultdict(dict)
        # Current and previous location
        self.current_location: str = self._extract_location(self.state.observation)
        self.previous_location: str = ""
        # Score tracking
        self.previous_score: int = 0

    def _extract_location(self, obs: str) -> str:
        for line in obs.strip().split("\n"):
            line = line.strip()
            if line and len(line) < 80:
                return line
        return "Unknown"

    def take_action(self, action: str) -> dict:
        """Execute action, return enriched result dict."""
        obs_before = self.state.observation
        score_before = self.state.score
        prev_location = self.current_location

        self.state = self.env.step(action)
        obs_after = self.state.observation
        score_after = self.state.score
        reward = self.state.reward

        # Track history
        self.history.append((action, obs_after))
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # Determine if action had any observable effect
        obs_changed = obs_before.strip() != obs_after.strip()
        score_changed = score_after > score_before

        # Update location BEFORE deciding if action failed
        # (a direction can produce same obs but still move to new room)
        new_location = self._extract_location(obs_after)
        location_changed = new_location != prev_location

        # An action is truly failed only if NOTHING changed at all:
        # same obs, same score, AND same location
        if not obs_changed and not score_changed and not location_changed:
            self.failed_actions[prev_location].add(action.lower())

        action_lower = action.lower().strip()

        DIRECTIONS = {"north","south","east","west","up","down","enter","exit",
                      "n","s","e","w","u","d","ne","nw","se","sw",
                      "northeast","northwest","southeast","southwest"}

        if action_lower in DIRECTIONS and new_location != prev_location:
            self.location_graph[prev_location][action_lower] = new_location

        self.previous_location = prev_location
        self.current_location = new_location
        self.previous_score = score_before

        return {
            "observation": obs_after,
            "score": score_after,
            "reward": reward,
            "moves": self.state.moves,
            "done": self.state.done,
            "obs_changed": obs_changed,
        }

    def get_memory(self) -> str:
        """Rich state summary including failed actions for current location."""
        failed_here = sorted(self.failed_actions.get(self.current_location, set()))
        recent = self.history[-5:]
        recent_str = "\n".join(
            f"  > {a} → {r.split(chr(10))[0][:70]}" for a, r in recent
        ) if recent else "  (none yet)"

        # Known exits
        exits = self.location_graph.get(self.current_location, {})
        exits_str = ", ".join(f"{d}→{dest}" for d, dest in exits.items()) if exits else "none mapped yet"

        failed_str = ", ".join(failed_here) if failed_here else "none"

        return (
            f"=== GAME STATE ===\n"
            f"Location:   {self.current_location}\n"
            f"Score:      {self.state.score} | Moves: {self.state.moves}\n"
            f"Game:       {self.game_name}\n\n"
            f"Known exits: {exits_str}\n"
            f"Failed actions HERE: {failed_str}\n\n"
            f"Recent history:\n{recent_str}\n\n"
            f"Current observation:\n{self.state.observation}"
        )

    def get_map(self) -> str:
        """Text map of explored locations."""
        if not self.location_graph:
            return "Map: No connections mapped yet. Try moving in different directions."

        lines = [f"Explored connections ({len(self.location_graph)} locations):"]
        for loc, exits in sorted(self.location_graph.items()):
            marker = " ← [YOU ARE HERE]" if loc == self.current_location else ""
            lines.append(f"\n  {loc}{marker}")
            for direction, dest in sorted(exits.items()):
                lines.append(f"    {direction} → {dest}")

        if self.current_location not in self.location_graph:
            lines.append(f"\n  {self.current_location} ← [YOU ARE HERE] (no exits mapped yet)")

        return "\n".join(lines)

    def get_inventory(self) -> str:
        """Clean inventory string."""
        items = self.state.inventory if (
            hasattr(self.state, 'inventory') and self.state.inventory
        ) else []

        if not items:
            return "You are empty-handed."

        names = []
        for item in items:
            item_str = str(item)
            # Jericho objects look like "sword: parent=..." -- extract name
            name = item_str.split(":")[0].strip() if ":" in item_str else item_str
            names.append(name)

        return "Carrying: " + ", ".join(names)


# Global game state
_game_state: GameState | None = None


def get_game() -> GameState:
    global _game_state
    if _game_state is None:
        game = os.environ.get("GAME", "lostpig")
        _game_state = GameState(game)
    return _game_state


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
def play_action(action: str) -> str:
    """
    Execute a game command and return the result.

    Args:
        action: Command to execute (e.g. 'north', 'take lamp', 'open mailbox')

    Returns:
        Game response including score info and whether the action had any effect.

    Valid commands:
        Movement: north, south, east, west, up, down, enter, exit (n/s/e/w/u/d)
        Objects:  take <item>, drop <item>, open <thing>, examine <thing>
        Other:    look, inventory, read <thing>, turn on lamp, wait
    """
    game = get_game()
    result = game.take_action(action)

    obs = result["observation"]
    score = result["score"]
    moves = result["moves"]
    reward = result["reward"]
    done = result["done"]
    obs_changed = result["obs_changed"]

    # Build informative suffix
    if reward > 0:
        suffix = f"\n\n✓ +{reward} points! (Total: {score} | Moves: {moves})"
    else:
        suffix = f"\n\n[Score: {score} | Moves: {moves}]"

    if not obs_changed and reward == 0:
        suffix += "\n[No effect — this action may not work here]"

    if done:
        suffix += "\n\nGAME OVER"

    return obs + suffix


@mcp.tool()
def memory() -> str:
    """
    Get current game state summary.

    Returns location, score, moves, known exits, failed actions at current
    location, and recent action history. Use this to plan your next move.
    """
    return get_game().get_memory()


@mcp.tool()
def get_map() -> str:
    """
    Get a map of explored locations and their connections.

    Useful for navigation: shows which directions lead where,
    and marks your current position.
    """
    return get_game().get_map()


#@mcp.tool()
#def inventory() -> str:
#    """Check what items you are currently carrying."""
#    return get_game().get_inventory()

# Dans mcp_server.py — corriger get_inventory()
@mcp.tool()
def inventory() -> str:
    """Check what items you are currently carrying."""
    game = get_game()
    # Appel direct au jeu pour avoir les vrais noms
    result = game.take_action("inventory")
    return result["observation"]


@mcp.tool()
def get_location() -> str:
    """
    Get the name of your current location.

    Faster than calling memory() when you only need the location name.
    """
    game = get_game()
    return f"Current location: {game.current_location}"


@mcp.tool()
def suggest_actions() -> str:
    """
    Get a prioritized list of recommended actions for the current situation.

    Returns directions not yet explored from this location, and objects
    mentioned in the current observation that haven't been examined yet.
    Use this when unsure what to do next — it avoids wasting moves on
    actions already known to fail.
    """
    game = get_game()
    obs = game.state.observation.lower()
    location = game.current_location
    failed = game.failed_actions.get(location, set())
    known_exits = set(game.location_graph.get(location, {}).keys())

    suggestions = []

    # Source of truth for exits: directions that have been mapped in location_graph
    # (these are directions that definitively WORKED — they changed location)
    confirmed_exits = set(game.location_graph.get(location, {}).keys())

    # Truly failed directions: in failed_actions AND not in location_graph
    # (if a direction is in location_graph, it worked at some point — never ban it)
    truly_failed = {
        d for d in game.failed_actions.get(location, set())
        if d not in confirmed_exits
    }

    # 1. Untried directions: not confirmed, not truly failed
    ALL_DIRECTIONS = [
        "north", "south", "east", "west", "up", "down",
        "northeast", "northwest", "southeast", "southwest", "enter", "exit"
    ]
    untried_dirs = [
        d for d in ALL_DIRECTIONS
        if d not in confirmed_exits and d not in truly_failed
    ]
    if untried_dirs:
        suggestions.append(f"Untried directions: {', '.join(untried_dirs[:6])}")

    # 2. Known working exits (already mapped — always show them)
    if confirmed_exits:
        exits_str = ", ".join(sorted(confirmed_exits))
        suggestions.append(f"Known exits (confirmed working): {exits_str}")

    # 3. Objects in observation not yet examined
    EXAMINABLE_KEYWORDS = [
        "mailbox", "sword", "lamp", "lantern", "torch", "rope", "key", "keys",
        "door", "trapdoor", "chest", "box", "bag", "note", "leaflet", "sign",
        "inscription", "button", "lever", "switch", "bottle", "book", "scroll",
        "statue", "fountain", "curtain", "rug", "table", "pedestal", "altar",
        "gnome", "troll", "thief", "adventurer", "skeleton", "ghost",
        "painting", "picture", "window", "grate", "ladder", "stairs",
        "crack", "hole", "passage", "tunnel",
    ]
    # Use truly_failed (not all failed_actions) so we don't over-exclude
    found_objects = [
        kw for kw in EXAMINABLE_KEYWORDS
        if kw in obs and f"examine {kw}" not in truly_failed
    ]
    if found_objects:
        suggestions.append(f"Objects to examine: {', '.join(f'examine {o}' for o in found_objects[:5])}")

    # 4. Inventory tip
    inv_result = game.take_action("inventory")
    inv_obs = inv_result["observation"].lower()
    if "torch" in inv_obs or "lantern" in inv_obs or "lamp" in inv_obs:
        if "turn on" not in truly_failed:
            suggestions.append("Tip: try 'turn on torch' or 'turn on lamp' if in dark area")

    if not suggestions:
        suggestions.append("No obvious suggestions — try 'look' then examine objects in description")

    return "=== SUGGESTED ACTIONS ===\n" + "\n".join(suggestions)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
