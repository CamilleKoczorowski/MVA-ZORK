"""
MCP Server V11 — Fast & Reliable Text Adventure Server

Key changes vs V10:
- REMOVED get_valid_actions (Jericho) — was the root cause of hangs
  (concurrent.futures thread timeout doesn't kill the thread → GIL blocked → server unresponsive)
- Server is now 100% non-blocking: only uses env.step() and env.get_player_location()
- Kept: BFS navigate_to_frontier, location_graph, failed_actions tracking
"""

import sys
import os
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv

INITIAL_GAME = os.environ.get("GAME", "lostpig")

mcp = FastMCP("Text Adventure Server V11")


# =============================================================================
# Game State
# =============================================================================

FAILURE_PHRASES = [
    "i don't understand", "i don't know the word",
    "you can't go that way", "you can't see any",
    "that's not something", "nothing happens", "you can't",
    "there is no", "i don't know how to", "that verb",
    "not open", "doesn't work", "[no effect",
    "doesn't seem to", "isn't something", "don't need to",
    "already", "huh?", "what?",
]

DIRECTIONS = frozenset([
    "north", "south", "east", "west", "up", "down",
    "northeast", "northwest", "southeast", "southwest",
    "enter", "exit", "n", "s", "e", "w", "u", "d",
    "ne", "nw", "se", "sw",
])


class GameState:
    def __init__(self, game: str = "lostpig"):
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()

        self.history: list[tuple[str, str]] = []
        self.failed_actions: dict[str, set[str]] = defaultdict(set)
        self.location_graph: dict[str, dict[str, str]] = defaultdict(dict)

        self.current_location: str = self._get_jericho_location()
        self.previous_location: str = ""
        self.previous_score: int = 0

    def _get_jericho_location(self) -> str:
        try:
            loc = str(self.env.env.get_player_location())
            if loc and loc != "None" and len(loc) < 80:
                return loc
        except Exception:
            pass
        return self._extract_location_from_text(self.state.observation)

    def _extract_location_from_text(self, obs: str) -> str:
        MSG = [
            "you ", "there ", "that ", "it ", "ok,", "oof",
            "already", "can't", "not ", "only ", "just ",
            "have", "hear", "look ", "see ", "but ", "this ",
            "nothing", "don't", "doesn't", "won't", "grunk",
        ]
        for line in obs.strip().split("\n"):
            line = line.strip()
            if not line or len(line) > 35:
                continue
            if not line[0].isupper():
                continue
            if line[-1] in ("!", "?", ".", ",", ";", ":"):
                continue
            if any(kw in line.lower() for kw in MSG):
                continue
            return line
        return "Unknown"

    def take_action(self, action: str) -> dict:
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

        obs_lower = obs_after.lower()
        is_explicit_failure = any(p in obs_lower for p in FAILURE_PHRASES)
        obs_changed = obs_before.strip() != obs_after.strip()
        score_changed = score_after > score_before
        new_location = self._get_jericho_location()
        location_changed = new_location != prev_location

        is_failed = is_explicit_failure or (
            not obs_changed and not score_changed and not location_changed
        )

        if is_failed and not prev_location.startswith("Unknown"):
            self.failed_actions[prev_location].add(action.lower())

        # Build location graph for BFS
        if (action.lower().strip() in DIRECTIONS and location_changed
                and not prev_location.startswith("Unknown")
                and not new_location.startswith("Unknown")):
            self.location_graph[prev_location][action.lower().strip()] = new_location
            # Record reverse direction
            REVERSE = {
                "north": "south", "south": "north",
                "east": "west", "west": "east",
                "up": "down", "down": "up",
                "northeast": "southwest", "southwest": "northeast",
                "northwest": "southeast", "southeast": "northwest",
                "enter": "exit", "exit": "enter",
            }
            rev = REVERSE.get(action.lower().strip())
            if rev:
                self.location_graph[new_location][rev] = prev_location

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
            "location_changed": location_changed,
            "is_failed": is_failed,
        }

    # -------------------------------------------------------------------------
    # BFS navigation
    # -------------------------------------------------------------------------

    CARDINAL = ["north", "south", "east", "west", "up", "down",
                "northeast", "northwest", "southeast", "southwest"]

    def compute_frontier_path(self) -> list[str]:
        start = self.current_location
        if start.startswith("Unknown"):
            return []

        graph = self.location_graph

        def is_frontier(loc: str) -> bool:
            confirmed = set(graph.get(loc, {}).keys())
            failed = self.failed_actions.get(loc, set())
            tried = confirmed | failed
            # Has untried cardinal directions
            return any(d not in tried for d in self.CARDINAL[:6])  # N/S/E/W/U/D

        queue: deque[tuple[str, list[str]]] = deque([(start, [])])
        visited: set[str] = {start}

        while queue:
            loc, path = queue.popleft()
            if loc != start and is_frontier(loc):
                return path
            for direction, dest in graph.get(loc, {}).items():
                if dest not in visited:
                    visited.add(dest)
                    queue.append((dest, path + [direction]))
        return []

    def get_memory(self) -> str:
        failed_here = sorted(self.failed_actions.get(self.current_location, set()))
        recent = self.history[-5:]
        recent_str = "\n".join(
            f"  > {a} → {r.split(chr(10))[0][:70]}" for a, r in recent
        ) if recent else "  (none yet)"
        exits = self.location_graph.get(self.current_location, {})
        exits_str = ", ".join(f"{d}→{dest}" for d, dest in exits.items()) if exits else "none mapped"
        failed_str = ", ".join(failed_here) if failed_here else "none"
        return (
            f"=== GAME STATE ===\n"
            f"Location:    {self.current_location}\n"
            f"Score:       {self.state.score} / {self.state.max_score} | Moves: {self.state.moves}\n"
            f"Game:        {self.game_name}\n\n"
            f"Known exits: {exits_str}\n"
            f"Failed here: {failed_str}\n\n"
            f"Recent:\n{recent_str}\n\n"
            f"Observation:\n{self.state.observation}"
        )

    def get_map(self) -> str:
        if not self.location_graph:
            return "Map: No connections mapped yet."
        lines = [f"Explored map ({len(self.location_graph)} locations):"]
        for loc, exits in sorted(self.location_graph.items()):
            marker = " ← [HERE]" if loc == self.current_location else ""
            lines.append(f"\n  {loc}{marker}")
            for direction, dest in sorted(exits.items()):
                lines.append(f"    {direction} → {dest}")
        if self.current_location not in self.location_graph:
            lines.append(f"\n  {self.current_location} ← [HERE] (no exits mapped)")
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
    """Execute a game command (e.g. 'north', 'take lamp', 'examine door')."""
    game = get_game()
    result = game.take_action(action)
    obs = result["observation"]
    score = result["score"]
    max_score = result["max_score"]
    moves = result["moves"]
    reward = result["reward"]
    done = result["done"]
    location = result["location"]
    is_failed = result["is_failed"]

    if reward > 0:
        suffix = f"\n\n✓ +{reward} points! (Total: {score}/{max_score} | Moves: {moves} | Location: {location})"
    else:
        suffix = f"\n\n[Score: {score}/{max_score} | Moves: {moves} | Location: {location}]"

    if is_failed:
        suffix += "\n[FAILED — action had no effect here]"

    if done:
        suffix += "\n\nGAME OVER"

    return obs + suffix


@mcp.tool()
def memory() -> str:
    """Get current game state summary including score, location, recent history."""
    return get_game().get_memory()


@mcp.tool()
def get_map() -> str:
    """Get explored map of locations and connections."""
    return get_game().get_map()


@mcp.tool()
def inventory() -> str:
    """Check what items you are carrying."""
    game = get_game()
    result = game.take_action("inventory")
    return result["observation"]


@mcp.tool()
def get_location() -> str:
    """Get current location name (uses Jericho API directly)."""
    game = get_game()
    return f"Current location: {game.current_location}"


@mcp.tool()
def suggest_actions() -> str:
    """Get suggested untried actions for current location."""
    game = get_game()
    location = game.current_location
    suggestions = []

    # Known exits from graph
    confirmed_exits = set(game.location_graph.get(location, {}).keys())
    failed = game.failed_actions.get(location, set())

    # Untried cardinal directions
    untried = [d for d in game.CARDINAL if d not in confirmed_exits and d not in failed]
    if untried:
        suggestions.append(f"Untried directions: {', '.join(untried)}")
    if confirmed_exits:
        suggestions.append(f"Confirmed exits: {', '.join(sorted(confirmed_exits))}")

    # BFS frontier
    path = game.compute_frontier_path()
    if path:
        suggestions.append(f"BFS: {len(path)} steps to nearest frontier → call navigate_to_frontier")
    else:
        suggestions.append("BFS: All reachable rooms seem explored.")

    # Failed actions
    if failed:
        suggestions.append(f"Failed here (do NOT repeat): {', '.join(sorted(failed))}")

    return "=== SUGGESTIONS ===\n" + "\n".join(suggestions)


@mcp.tool()
def navigate_to_frontier() -> str:
    """BFS to nearest room with unexplored exits. Returns directions to execute."""
    game = get_game()
    path = game.compute_frontier_path()
    if not path:
        return (
            "NO FRONTIER: All reachable rooms explored.\n"
            "Try: examine objects, look for hidden exits (up, down, enter)."
        )
    return (
        f"FRONTIER ({len(path)} steps):\n"
        + "\n".join(path)
        + "\n\nExecute each direction with play_action."
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
