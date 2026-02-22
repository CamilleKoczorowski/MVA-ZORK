"""
MCP Server V11c — Fast, no Jericho get_valid_actions

Changes vs V11:
- Reverse directions in location_graph for better BFS
- Cleaner failure detection
"""

import sys
import os
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv

mcp = FastMCP("Text Adventure Server V11c")

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

REVERSE = {
    "north": "south", "south": "north",
    "east": "west", "west": "east",
    "up": "down", "down": "up",
    "northeast": "southwest", "southwest": "northeast",
    "northwest": "southeast", "southeast": "northwest",
    "enter": "exit", "exit": "enter",
}


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

        if (action.lower().strip() in DIRECTIONS and location_changed
                and not prev_location.startswith("Unknown")
                and not new_location.startswith("Unknown")):
            self.location_graph[prev_location][action.lower().strip()] = new_location
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
            return any(d not in tried for d in self.CARDINAL[:6])

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
            f"Location:    {self.current_location}\n"
            f"Score:       {self.state.score} / {self.state.max_score} | Moves: {self.state.moves}\n"
            f"Known exits: {exits_str}\n"
            f"Failed here: {failed_str}\n"
            f"Recent:\n{recent_str}\n"
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


_game_state: GameState | None = None

def get_game() -> GameState:
    global _game_state
    if _game_state is None:
        game = os.environ.get("GAME", "lostpig")
        _game_state = GameState(game)
    return _game_state


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
    """Get current game state summary."""
    return get_game().get_memory()

@mcp.tool()
def get_map() -> str:
    """Get explored map."""
    return get_game().get_map()

@mcp.tool()
def inventory() -> str:
    """Check items you carry."""
    game = get_game()
    result = game.take_action("inventory")
    return result["observation"]

@mcp.tool()
def get_location() -> str:
    """Get current location name."""
    game = get_game()
    return f"Current location: {game.current_location}"

@mcp.tool()
def suggest_actions() -> str:
    """Get untried actions for current location."""
    game = get_game()
    location = game.current_location
    suggestions = []
    confirmed = set(game.location_graph.get(location, {}).keys())
    failed = game.failed_actions.get(location, set())
    untried = [d for d in game.CARDINAL if d not in confirmed and d not in failed]
    if untried:
        suggestions.append(f"Untried directions: {', '.join(untried)}")
    if confirmed:
        suggestions.append(f"Confirmed exits: {', '.join(sorted(confirmed))}")
    path = game.compute_frontier_path()
    if path:
        suggestions.append(f"BFS: {len(path)} steps to frontier")
    if failed:
        suggestions.append(f"Failed: {', '.join(sorted(failed))}")
    return "\n".join(suggestions)

@mcp.tool()
def navigate_to_frontier() -> str:
    """BFS to nearest room with unexplored exits."""
    game = get_game()
    path = game.compute_frontier_path()
    if not path:
        return "NO FRONTIER: All reachable rooms explored. Try: examine, up, down, enter."
    return f"FRONTIER ({len(path)} steps):\n" + "\n".join(path)


if __name__ == "__main__":
    mcp.run()
