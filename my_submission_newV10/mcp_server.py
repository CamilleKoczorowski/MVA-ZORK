"""
MCP Server V10 — Text Adventure Games

Architecture redesign following professor's hints:
- NEW: get_valid_actions() tool — exposes Jericho's valid action list directly
- NEW: get_location_state() — structured per-location info (valid exits, objects)
- KEEP: navigate_to_frontier() BFS, location_graph, failed_actions
- KEEP: All V7-V9 fixes (obs-hash, Unknown filtering, 35-char limit)
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

    # -------------------------------------------------------------------------
    # Location extraction
    # -------------------------------------------------------------------------

    def _get_jericho_location(self) -> str:
        try:
            loc = str(self.env.env.get_player_location())
            if loc and loc != "None" and len(loc) < 80:
                return loc
        except Exception:
            pass
        return self._extract_location_from_text(self.state.observation)

    def _extract_location_from_text(self, obs: str) -> str:
        MESSAGE_INDICATORS = [
            "you ", "there ", "that ", "it ", "ok,", "oof",
            "already", "can't", "not ", "only ", "just ",
            "have", "hear", "look ", "see ", "but ", "this ",
            "nothing", "don't", "doesn't", "won't",
        ]
        for line in obs.strip().split("\n"):
            line = line.strip()
            if not line or len(line) > 35:
                continue
            if not line[0].isupper():
                continue
            if line[-1] in ("!", "?", ".", ",", ";", ":"):
                continue
            if any(kw in line.lower() for kw in MESSAGE_INDICATORS):
                continue
            return line
        return "Unknown"

    # -------------------------------------------------------------------------
    # Action execution
    # -------------------------------------------------------------------------

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

        obs_changed = obs_before.strip() != obs_after.strip()
        score_changed = score_after > score_before
        new_location = self._get_jericho_location()
        location_changed = new_location != prev_location

        if not obs_changed and not score_changed and not location_changed:
            if not prev_location.startswith("Unknown"):
                self.failed_actions[prev_location].add(action.lower())

        DIRECTIONS = {
            "north","south","east","west","up","down","enter","exit",
            "n","s","e","w","u","d","ne","nw","se","sw",
            "northeast","northwest","southeast","southwest",
        }
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
            "location_changed": location_changed,
        }

    # -------------------------------------------------------------------------
    # Jericho valid actions — KEY NEW FEATURE (professor hint #2)
    # -------------------------------------------------------------------------

    def _fallback_actions(self) -> list[str]:
        """Fallback when Jericho get_valid_actions fails: return plausible directions."""
        CARDINAL = ["north", "south", "east", "west", "up", "down",
                    "northeast", "northwest", "southeast", "southwest"]
        failed = self.failed_actions.get(self.current_location, set())
        return [d for d in CARDINAL if d not in failed]

    def get_jericho_valid_actions(self, timeout_sec: float = 1.5) -> list[str]:
        """
        Ask Jericho for valid actions in current state.
        Uses a thread + timeout to avoid hanging (Jericho brute-forces candidates).
        Falls back to cardinal directions if timeout or error.
        """
        import sys, io, concurrent.futures
        SKIP = {"", "yes", "no", "again", "oops", "undo", "restart", "quit", "score", "verbose"}

        def _call():
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                # Prof's pattern first
                try:
                    v = self.env.get_valid_actions()
                    if v:
                        return v
                except Exception:
                    pass
                # Fallback: inner env
                return self.env.env.get_valid_actions()
            finally:
                sys.stderr = old_stderr

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                valid = future.result(timeout=timeout_sec)
            if valid:
                return [a for a in valid if a.lower() not in SKIP]
        except concurrent.futures.TimeoutError:
            pass  # Jericho hung → use fallback
        except Exception:
            pass
        return self._fallback_actions()

    # -------------------------------------------------------------------------
    # BFS navigation
    # -------------------------------------------------------------------------

    ALL_DIRECTIONS = frozenset([
        "north","south","east","west","up","down",
        "northeast","northwest","southeast","southwest","enter","exit",
    ])

    def compute_frontier_path(self) -> list[str]:
        start = self.current_location
        graph = self.location_graph
        CARDINAL = frozenset(["north", "south", "east", "west", "up", "down"])

        def is_frontier(loc: str) -> bool:
            confirmed = set(graph.get(loc, {}).keys())
            failed = self.failed_actions.get(loc, set())
            truly_failed = failed - confirmed
            tried = confirmed | truly_failed
            cardinal_untried = CARDINAL - tried
            return len(cardinal_untried) > 0

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
            return "Map: No connections mapped yet."
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
    """Get current game state summary."""
    return get_game().get_memory()


@mcp.tool()
def get_map() -> str:
    """Get a map of explored locations and their connections."""
    return get_game().get_map()


@mcp.tool()
def inventory() -> str:
    """Check what items you are currently carrying."""
    game = get_game()
    result = game.take_action("inventory")
    return result["observation"]


@mcp.tool()
def get_location() -> str:
    """Get the name of your current location (reliable, uses Jericho directly)."""
    game = get_game()
    return f"Current location: {game.current_location}"


@mcp.tool()
def get_valid_actions() -> str:
    """
    Get the list of valid actions in the current game state, directly from Jericho.

    This is the authoritative list — no guessing required. Use this on entering
    a new location to know exactly what directions and interactions are possible.

    Returns a newline-separated list of valid actions.
    """
    game = get_game()
    valid = game.get_jericho_valid_actions()
    if not valid:
        return "No valid actions returned (Jericho API unavailable — use manual exploration)."
    lines = [f"Valid actions at '{game.current_location}' ({len(valid)} total):"]
    # Separate directions from object interactions for clarity
    DIRS = {"north","south","east","west","up","down","northeast","northwest",
            "southeast","southwest","enter","exit","n","s","e","w","u","d"}
    directions = [a for a in valid if a.lower() in DIRS]
    others = [a for a in valid if a.lower() not in DIRS]
    if directions:
        lines.append(f"  Directions: {', '.join(directions)}")
    if others:
        lines.append(f"  Actions: {', '.join(others[:20])}")  # cap at 20 to avoid token flood
    return "\n".join(lines)


@mcp.tool()
def suggest_actions() -> str:
    """
    Get recommended actions: valid directions + objects in observation.
    Combines Jericho valid actions with BFS frontier hint.
    """
    game = get_game()
    obs = game.state.observation.lower()
    location = game.current_location

    suggestions = []

    # 1. Valid actions from Jericho (authoritative)
    valid = game.get_jericho_valid_actions()
    DIRS = {"north","south","east","west","up","down","northeast","northwest",
            "southeast","southwest","enter","exit","n","s","e","w","u","d"}
    if valid:
        confirmed_dirs = [a for a in valid if a.lower() in DIRS]
        confirmed_exits = set(game.location_graph.get(location, {}).keys())
        failed = game.failed_actions.get(location, set())
        untried_dirs = [d for d in confirmed_dirs if d not in confirmed_exits and d not in failed]
        if untried_dirs:
            suggestions.append(f"Untried valid directions: {', '.join(untried_dirs)}")
        if confirmed_exits:
            suggestions.append(f"Confirmed exits: {', '.join(sorted(confirmed_exits))}")
        object_actions = [a for a in valid if a.lower() not in DIRS and a not in failed]
        if object_actions:
            suggestions.append(f"Valid object actions: {', '.join(object_actions[:10])}")
    else:
        # Fallback if Jericho unavailable
        confirmed_exits = set(game.location_graph.get(location, {}).keys())
        failed = game.failed_actions.get(location, set())
        ALL_DIRS = ["north","south","east","west","up","down","northeast","northwest","southeast","southwest"]
        untried = [d for d in ALL_DIRS if d not in confirmed_exits and d not in failed]
        if untried:
            suggestions.append(f"Untried directions: {', '.join(untried[:8])}")

    # 2. BFS frontier hint
    path = game.compute_frontier_path()
    if path:
        suggestions.append(f"BFS FRONTIER: call navigate_to_frontier ({len(path)} steps to nearest unexplored area)")
    else:
        suggestions.append("BFS: All reachable named rooms seem fully explored.")

    return "=== SUGGESTED ACTIONS ===\n" + "\n".join(suggestions)


@mcp.tool()
def navigate_to_frontier() -> str:
    """
    BFS path to nearest frontier location (room with unexplored exits).
    Returns directions to execute one by one with play_action.
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
        f"{steps}\n\nExecute each direction with play_action, one at a time."
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
