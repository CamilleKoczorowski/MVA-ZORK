"""
MCP Server V4 — Text Adventure Games

Changes vs V3-bis:
- FIX: _extract_location now uses Jericho's get_player_location() as primary source,
       with a safer text-based fallback that filters out error messages.
- FIX: suggest_actions no longer calls take_action("inventory") (side effect/move).
- NEW: navigate_to_frontier() tool — BFS to the nearest unexplored frontier node.
- NEW: max_score exposed in memory() and play_action() output.
"""

import sys
import os
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv

INITIAL_GAME = os.environ.get("GAME", "lostpig")

mcp = FastMCP("Text Adventure Server - V4")


# =============================================================================
# Game State
# =============================================================================

class GameState:
    """
    Extended game state with per-location failed action tracking.
    V4: uses Jericho location as primary source, safer text fallback.
    """

    def __init__(self, game: str = "lostpig"):
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()

        self.history: list[tuple[str, str]] = []
        self.failed_actions: dict[str, set[str]] = defaultdict(set)
        self.location_graph: dict[str, dict[str, str]] = defaultdict(dict)

        # V4: use Jericho location as primary
        self.current_location: str = self._get_jericho_location()
        self.previous_location: str = ""
        self.previous_score: int = 0

    # -------------------------------------------------------------------------
    # Location extraction — V4: Jericho primary, text fallback
    # -------------------------------------------------------------------------

    def _get_jericho_location(self) -> str:
        """Use Jericho's get_player_location() — reliable, no text parsing."""
        try:
            loc = str(self.env.env.get_player_location())
            # Jericho returns e.g. "Outside" or "Fountain Room"
            if loc and loc != "None" and len(loc) < 80:
                return loc
        except Exception:
            pass
        return self._extract_location_from_text(self.state.observation)

    def _extract_location_from_text(self, obs: str) -> str:
        """
        Safe text fallback: a location name is short, starts with uppercase,
        does not end with ! / ? and does not look like a game message.
        """
        # Keywords that appear in error/game messages but NOT in room titles
        MESSAGE_INDICATORS = [
            "you ", "there ", "that ", "it ", "ok,", "oof",
            "already", "can't", "not ", "only ", "just ",
            "have", "hear", "look ", "see ", "but ", "this ",
            "nothing", "don't", "doesn't", "won't",
        ]
        for line in obs.strip().split("\n"):
            line = line.strip()
            if not line or len(line) >= 60:
                continue
            if not line[0].isupper():
                continue
            if line.endswith("!") or line.endswith("?"):
                continue
            line_lower = line.lower()
            if any(kw in line_lower for kw in MESSAGE_INDICATORS):
                continue
            # Looks like a title
            return line
        return "Unknown"

    # -------------------------------------------------------------------------
    # Action execution
    # -------------------------------------------------------------------------

    def take_action(self, action: str) -> dict:
        """Execute action, return enriched result dict."""
        obs_before = self.state.observation
        score_before = self.state.score
        prev_location = self.current_location

        self.state = self.env.step(action)
        obs_after = self.state.observation
        score_after = self.state.score
        reward = self.state.reward

        self.history.append((action, obs_after))
        if len(self.history) > 100:
            self.history = self.history[-100:]

        obs_changed = obs_before.strip() != obs_after.strip()
        score_changed = score_after > score_before

        # V4: primary location source from Jericho
        new_location = self._get_jericho_location()
        location_changed = new_location != prev_location

        # Action failed only if obs + score + location all unchanged
        if not obs_changed and not score_changed and not location_changed:
            self.failed_actions[prev_location].add(action.lower())

        DIRECTIONS = {
            "north","south","east","west","up","down","enter","exit",
            "n","s","e","w","u","d","ne","nw","se","sw",
            "northeast","northwest","southeast","southwest",
        }
        # Only record transitions between two known (non-Unknown) locations
        if (action.lower().strip() in DIRECTIONS and location_changed
                and not prev_location.startswith("Unknown")
                and not new_location.startswith("Unknown")):
            self.location_graph[prev_location][action.lower().strip()] = new_location

        self.previous_location = prev_location
        self.current_location = new_location
        self.previous_score = score_before

        return {
            "observation": obs_after,
            "score": score_after,
            "max_score": self.state.max_score,
            "reward": reward,
            "moves": self.state.moves,
            "done": self.state.done,
            "obs_changed": obs_changed,
            "location": new_location,
        }

    # -------------------------------------------------------------------------
    # BFS navigation (V4 core feature)
    # -------------------------------------------------------------------------

    ALL_DIRECTIONS = frozenset([
        "north","south","east","west","up","down",
        "northeast","northwest","southeast","southwest","enter","exit",
    ])

    def compute_frontier_path(self) -> list[str]:
        """
        BFS in location_graph to find the shortest path to the nearest
        'frontier' node — a visited location that still has untried directions.

        Returns the list of directions to follow, or [] if none found.
        """
        start = self.current_location
        graph = self.location_graph

        CARDINAL = frozenset(["north", "south", "east", "west", "up", "down"])

        def is_frontier(loc: str) -> bool:
            confirmed = set(graph.get(loc, {}).keys())
            failed = self.failed_actions.get(loc, set())
            # truly_failed = directions tried with no effect at this specific location
            truly_failed = failed - confirmed
            tried = confirmed | truly_failed
            # Only consider cardinal directions as frontier indicators —
            # diagonal/enter/exit are rare, so "untried diagonal" ≠ real frontier
            cardinal_untried = CARDINAL - tried
            return len(cardinal_untried) > 0

        queue: deque[tuple[str, list[str]]] = deque([(start, [])])
        visited: set[str] = {start}

        while queue:
            loc, path = queue.popleft()
            # Any node other than start that is a frontier counts
            if loc != start and is_frontier(loc):
                return path
            for direction, dest in graph.get(loc, {}).items():
                if dest not in visited:
                    visited.add(dest)
                    queue.append((dest, path + [direction]))

        return []  # No frontier reachable

    # -------------------------------------------------------------------------
    # Summary helpers
    # -------------------------------------------------------------------------

    def get_memory(self) -> str:
        failed_here = sorted(self.failed_actions.get(self.current_location, set()))
        recent = self.history[-5:]
        recent_str = "\n".join(
            f"  > {a} → {r.split(chr(10))[0][:70]}" for a, r in recent
        ) if recent else "  (none yet)"

        exits = self.location_graph.get(self.current_location, {})
        exits_str = ", ".join(f"{d}→{dest}" for d, dest in exits.items()) if exits else "none mapped yet"
        failed_str = ", ".join(failed_here) if failed_here else "none"

        return (
            f"=== GAME STATE ===\n"
            f"Location:    {self.current_location}\n"
            f"Score:       {self.state.score} / {self.state.max_score} | Moves: {self.state.moves}\n"
            f"Game:        {self.game_name}\n\n"
            f"Known exits: {exits_str}\n"
            f"Failed here: {failed_str}\n\n"
            f"Recent history:\n{recent_str}\n\n"
            f"Current observation:\n{self.state.observation}"
        )

    def get_map(self) -> str:
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
    max_score = result["max_score"]
    moves = result["moves"]
    reward = result["reward"]
    done = result["done"]
    obs_changed = result["obs_changed"]
    location = result["location"]

    if reward > 0:
        suffix = f"\n\n✓ +{reward} points! (Total: {score}/{max_score} | Moves: {moves} | Location: {location})"
    else:
        suffix = f"\n\n[Score: {score}/{max_score} | Moves: {moves} | Location: {location}]"

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


@mcp.tool()
def inventory() -> str:
    """Check what items you are currently carrying."""
    game = get_game()
    result = game.take_action("inventory")
    return result["observation"]


@mcp.tool()
def get_location() -> str:
    """
    Get the name of your current location (reliable, uses Jericho directly).
    """
    game = get_game()
    return f"Current location: {game.current_location}"


@mcp.tool()
def suggest_actions() -> str:
    """
    Get a prioritized list of recommended actions for the current situation.

    Returns untried directions and objects mentioned in the current observation
    that haven't been examined yet.
    NOTE: does NOT execute any game action (no side effects).
    """
    game = get_game()
    obs = game.state.observation.lower()
    location = game.current_location

    confirmed_exits = set(game.location_graph.get(location, {}).keys())
    failed = game.failed_actions.get(location, set())
    truly_failed = failed - confirmed_exits

    suggestions = []

    # 1. Untried directions
    ALL_DIRS = [
        "north", "south", "east", "west", "up", "down",
        "northeast", "northwest", "southeast", "southwest", "enter", "exit",
    ]
    untried = [d for d in ALL_DIRS if d not in confirmed_exits and d not in truly_failed]
    if untried:
        suggestions.append(f"Untried directions: {', '.join(untried[:8])}")

    # 2. Known working exits
    if confirmed_exits:
        suggestions.append(f"Confirmed exits: {', '.join(sorted(confirmed_exits))}")

    # 3. Objects to examine
    EXAMINABLE = [
        "mailbox","sword","lamp","lantern","torch","rope","key","keys","door",
        "trapdoor","chest","box","bag","note","leaflet","sign","inscription",
        "button","lever","switch","bottle","book","scroll","statue","fountain",
        "curtain","rug","table","pedestal","altar","painting",
        "window","grate","ladder","crack","hole","passage","tunnel",
    ]
    found = [kw for kw in EXAMINABLE if kw in obs and f"examine {kw}" not in truly_failed]
    if found:
        suggestions.append(f"Objects to examine: {', '.join(f'examine {o}' for o in found[:6])}")

    # 4. BFS frontier hint (V4: no full path, just existence check)
    path = game.compute_frontier_path()
    if path:
        suggestions.append(
            f"BFS FRONTIER: call navigate_to_frontier to get path "
            f"({len(path)} steps to nearest unexplored area)"
        )
    else:
        suggestions.append("BFS: All reachable locations seem fully explored.")

    if not suggestions:
        suggestions.append("No obvious suggestions — try 'look' then examine objects.")

    return "=== SUGGESTED ACTIONS ===\n" + "\n".join(suggestions)


@mcp.tool()
def navigate_to_frontier() -> str:
    """
    Compute the shortest path (BFS) from your current location to the nearest
    'frontier' location — a visited room that still has unexplored exits.

    Returns a JSON-like sequence of directions to follow mechanically.
    The agent should execute each direction in order WITHOUT calling the LLM
    in between, to avoid wasting tokens on navigation.

    Returns:
        A newline-separated list of directions, or a message if no frontier found.

    Example output:
        FRONTIER PATH (3 steps):
        south
        east
        north
        Execute these directions one by one with play_action.
    """
    game = get_game()
    path = game.compute_frontier_path()

    if not path:
        return (
            "NO FRONTIER FOUND: All reachable locations appear fully explored.\n"
            "Try: examine objects more carefully, or look for hidden exits (up, down, enter)."
        )

    steps = "\n".join(path)
    return (
        f"FRONTIER PATH ({len(path)} steps to nearest unexplored area):\n"
        f"{steps}\n\n"
        f"Execute each direction with play_action, one at a time."
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
