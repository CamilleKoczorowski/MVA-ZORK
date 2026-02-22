"""
Student Agent V11e — Reliable location tracking + object interaction

KEY CHANGES vs V11d:
1. Server now keeps prev_location on failed actions → no more "Unknown" pollution
   → room keys are stable → failed_actions properly tracked → no loops
2. Room key = server's location name (stable for named rooms)
   For Unknown: hash of clean obs first 2 lines (same as V11d but fewer unknowns now)
3. After directions are exhausted: try "look", "examine" things from obs, "take" items
4. LLM prompt emphasizes object interaction for scoring
5. Removed reverse edge inference from server (non-Euclidean game maps)
"""

import asyncio
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
_hf_token = os.getenv("HF_TOKEN")
if not _hf_token:
    raise ValueError("HF_TOKEN not found in environment.")
LLM_CLIENT = InferenceClient(token=_hf_token)


def call_llm(prompt: str, system: str, seed: int, max_tokens: int = 400) -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    resp = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL, messages=messages,
        temperature=0.0, max_tokens=max_tokens, seed=seed,
    )
    return resp.choices[0].message.content


# =============================================================================
# Constants
# =============================================================================

CARDINAL_6 = ["north", "south", "east", "west", "up", "down"]
CARDINAL_10 = CARDINAL_6 + ["northeast", "northwest", "southeast", "southwest"]
ALL_MOVE = set(CARDINAL_10 + ["enter", "exit", "n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw"])

FAILURE_PHRASES = [
    "i don't understand", "i don't know the word",
    "you can't go that way", "you can't see any",
    "that's not something", "nothing happens", "you can't",
    "there is no", "i don't know how to", "that verb",
    "not open", "doesn't work", "[no effect",
    "doesn't seem to", "isn't something", "don't need to",
    "already", "huh?", "what?", "[failed",
    "not see any place", "only see way to go",
    "not leave without pig",
]

BAD_VERBS = {
    "check": "examine", "inspect": "examine", "search": "look",
    "grab": "take", "pick": "take", "use": "examine",
    "investigate": "examine", "get": "take", "go": "north",
    "hit": "attack", "kill": "attack",
}

REVERSE_DIR = {
    "north": "south", "south": "north", "east": "west", "west": "east",
    "up": "down", "down": "up", "northeast": "southwest",
    "southwest": "northeast", "northwest": "southeast",
    "southeast": "northwest", "enter": "exit", "exit": "enter",
}


def _clean_obs(text: str) -> str:
    """Strip tool metadata from observation."""
    for marker in ["\n\n✓ ", "\n\n[Score:", "\n\n[FAILED"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def _room_key(location: str, obs: str) -> str:
    """Room key: named rooms use name, Unknown uses clean obs hash."""
    if location != "Unknown":
        return location
    clean = _clean_obs(obs)
    lines = [l.strip() for l in clean.split("\n") if l.strip()][:2]
    return f"Unknown:{hash('|'.join(lines)) & 0xFFFFFF}"


def _extract_nouns_from_obs(obs: str) -> list[str]:
    """Extract interactable nouns from observation text."""
    clean = _clean_obs(obs).lower()
    # Common interactable objects in text adventures
    nouns = []
    # Pattern: "there is/are a/an/the NOUN"
    for m in re.finditer(r'(?:there|see|have|find)\s+(?:a|an|the|some)\s+(\w+(?:\s+\w+)?)', clean):
        nouns.append(m.group(1))
    # Pattern: common game nouns
    GAME_NOUNS = [
        "door", "key", "lamp", "torch", "sword", "coin", "box", "chest",
        "book", "scroll", "bottle", "rope", "ring", "stone", "lever",
        "button", "switch", "handle", "fountain", "statue", "picture",
        "shelf", "shelfs", "wall", "crack", "stream", "stairs",
        "pig", "gnome", "troll",
    ]
    for noun in GAME_NOUNS:
        if noun in clean:
            nouns.append(noun)
    return list(dict.fromkeys(nouns))  # dedupe preserving order


# =============================================================================
# Per-location state
# =============================================================================

@dataclass
class LocationState:
    key: str
    tried: set = field(default_factory=set)
    failed: set = field(default_factory=set)
    outcomes: dict = field(default_factory=dict)
    promising: deque = field(default_factory=deque)
    steps_here: int = 0
    escape_attempts: int = 0
    initialized: bool = False
    directions_done: bool = False  # all directions tried


@dataclass
class RunResult:
    final_score: int
    max_score: int
    moves: int
    locations_visited: set
    game_completed: bool
    error: Optional[str] = None
    history: list = field(default_factory=list)


# =============================================================================
# Agent Memory
# =============================================================================

class AgentMemory:
    def __init__(self):
        self.visited_locations: set = set()
        self.location_graph: dict = defaultdict(dict)
        self.action_history: list = []
        self.current_location: str = "Unknown"
        self._no_score_steps: int = 0

    def update(self, loc_before, loc_after, action, score_before, score_after):
        self.visited_locations.add(loc_before)
        self.visited_locations.add(loc_after)
        self.action_history.append(action)
        self._no_score_steps = 0 if score_after > score_before else self._no_score_steps + 1
        self.current_location = loc_after

    def is_stagnant(self, threshold=12):
        return self._no_score_steps >= threshold


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert text adventure game player. Maximize score and explore rooms.

RESPONSE FORMAT (strict):
THOUGHT: <1-2 sentences>
TOOL: <tool_name>
ARGS: {"key": "value"}

GAME COMMANDS for play_action:
- Directions: north, south, east, west, up, down, northeast, southeast, etc.
- Objects: take <item>, drop <item>, open <thing>, examine <thing>, read <thing>
- Other: look, inventory, wait, enter, exit

FORBIDDEN verbs: check, inspect, search, grab, use, help, go, investigate, get

TOOLS:
- play_action: execute game command
- navigate_to_frontier: BFS to nearest unexplored room
- memory: game state summary
- get_map: explored map
- inventory: check items

SCORING STRATEGY:
1. Try all valid directions to discover rooms
2. EXAMINE notable objects (fountain, statue, picture, shelf, etc.)
3. TAKE every item you can
4. INTERACT with creatures (talk to pig, examine gnome)
5. If stuck: navigate_to_frontier

RULES:
- NEVER repeat a FORBIDDEN action
- After exploring directions, INTERACT with objects to score points
- Pick up ALL items"""

PROMISING_SYSTEM = "Output ONLY a JSON array. No explanation."
PROMISING_PROMPT = """Text adventure room. Extract promising actions to try.
Priority order:
1. Directions not yet tried
2. "take <item>" for any visible items
3. "examine <object>" for notable objects
4. "open <thing>" for containers/doors

Already tried/failed: {tried}
Observation: {obs}

Return JSON array, e.g.: ["east", "take torch", "examine fountain", "open chest"]"""


# =============================================================================
# Student Agent
# =============================================================================

class StudentAgent:
    MAX_STEPS_ROOM = 10
    MAX_FAILS_ESCAPE = 3
    MAX_ESCAPE_PER_ROOM = 3

    def __init__(self):
        self.memory = AgentMemory()
        self._last_obs = ""
        self._last_score = 0
        self._max_score = 7
        self._autopilot: deque = deque()
        self._bfs_repeat = 0
        self._last_bfs_path: list = []
        self._bfs_cooldown = 0
        self._loc_states: dict = {}
        self._current_key = ""
        self._consec_fails = 0
        self._last_dir_in: str = ""

    def _get_loc(self, key: str) -> LocationState:
        if key not in self._loc_states:
            self._loc_states[key] = LocationState(key=key)
        return self._loc_states[key]

    def _init_room(self, key: str, obs: str, seed: int, verbose: bool):
        loc = self._get_loc(key)
        if loc.initialized:
            return
        loc.initialized = True
        clean = _clean_obs(obs)
        all_done = loc.failed | loc.tried

        # 1. Queue directions first
        for d in CARDINAL_10:
            if d not in all_done:
                loc.promising.append(d)

        # 2. Extract nouns and queue interactions
        nouns = _extract_nouns_from_obs(clean)
        for noun in nouns:
            for verb in ["examine", "take"]:
                action = f"{verb} {noun}"
                if action not in all_done:
                    loc.promising.append(action)

        # 3. LLM extract (more diverse actions)
        llm_acts = self._llm_extract(clean, all_done, seed)
        for a in llm_acts:
            if a not in all_done and a not in set(loc.promising):
                loc.promising.append(a)

        # 4. Always try "look" if not done
        if "look" not in all_done:
            loc.promising.append("look")

        if verbose:
            print(f"  [INIT '{key}'] promising={list(loc.promising)[:10]}...")

    def _llm_extract(self, obs: str, already: set, seed: int) -> list:
        try:
            prompt = PROMISING_PROMPT.format(tried=sorted(already)[:20], obs=obs[:600])
            raw = call_llm(prompt, PROMISING_SYSTEM, seed=seed, max_tokens=200)
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                actions = json.loads(m.group(0))
                return [a.lower().strip() for a in actions
                        if isinstance(a, str) and a.strip() and len(a) < 40]
        except Exception:
            pass
        return []

    def _is_fail(self, obs: str) -> bool:
        obs_lower = _clean_obs(obs).lower()
        return any(p in obs_lower for p in FAILURE_PHRASES)

    # ── Main run loop ────────────────────────────────────────────────────

    async def run(self, client, game: str, max_steps: int, seed: int,
                  verbose=False) -> RunResult:
        history = []
        moves = 0

        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        has_get_loc = "get_location" in tool_names
        has_nav = "navigate_to_frontier" in tool_names

        # Initial look
        r = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract(r)
        self._last_obs = observation
        location = await self._get_location(client, has_get_loc, observation)
        self.memory.visited_locations.add(location)

        if verbose:
            print(f"\n=== START ===\n{_clean_obs(observation)}\n[Location: {location}]")

        self._current_key = _room_key(location, observation)
        self._init_room(self._current_key, observation, seed, verbose)

        for step in range(1, max_steps + 1):
            loc = self._get_loc(self._current_key)
            loc.steps_here += 1

            if self._bfs_cooldown > 0:
                self._bfs_cooldown -= 1

            # ── 1. AUTOPILOT ────────────────────────────────────────────
            if self._autopilot:
                action = self._autopilot.popleft()
                new_obs, new_score, new_loc = await self._do(
                    client, action, location, has_get_loc)
                moves += 1

                if verbose:
                    print(f"\n[{step}|AUTO] {action} → {new_loc} | {_clean_obs(new_obs)[:80]}")

                if self._is_fail(new_obs):
                    self._autopilot.clear()
                    self._bfs_repeat += 1

                if not self._autopilot:
                    self._bfs_cooldown = 6

                self._handle_room_change(new_loc, new_obs, action, seed + step, verbose)
                location, observation = new_loc, new_obs
                self._last_obs, self._last_score = new_obs, new_score
                history.append(("AUTO", action, _clean_obs(new_obs)[:80]))
                if self._is_done(new_obs):
                    break
                continue

            # ── 2. CHECK ROOM KEY ────────────────────────────────────────
            new_key = _room_key(location, observation)
            if new_key != self._current_key:
                self._current_key = new_key
                self._consec_fails = 0
                self._init_room(new_key, observation, seed + step, verbose)
                loc = self._get_loc(new_key)

            # ── 3. ESCAPE after consecutive fails ────────────────────────
            if self._consec_fails >= self.MAX_FAILS_ESCAPE and loc.escape_attempts < self.MAX_ESCAPE_PER_ROOM:
                escape = self._escape_action(location, loc)
                loc.escape_attempts += 1
                if verbose:
                    print(f"\n[{step}|ESCAPE] {self._consec_fails} fails → {escape}")
                new_obs, new_score, new_loc = await self._do(
                    client, escape, location, has_get_loc)
                moves += 1
                self._consec_fails = 0
                self._handle_room_change(new_loc, new_obs, escape, seed + step, verbose)
                location, observation = new_loc, new_obs
                self._last_obs, self._last_score = new_obs, new_score
                history.append(("ESCAPE", escape, _clean_obs(new_obs)[:80]))
                if self._is_done(new_obs):
                    break
                continue

            # ── 4. PROMISING QUEUE ───────────────────────────────────────
            action_done = False
            while loc.promising:
                action = loc.promising.popleft()
                if action in loc.tried or action in loc.failed:
                    continue

                new_obs, new_score, new_loc = await self._do(
                    client, action, location, has_get_loc)
                moves += 1
                loc.tried.add(action)

                is_fail = self._is_fail(new_obs)
                if is_fail:
                    loc.failed.add(action)
                    self._consec_fails += 1
                else:
                    self._consec_fails = 0

                loc.outcomes[action] = _clean_obs(new_obs).split("\n")[0][:80]

                if verbose:
                    tag = "FAIL" if is_fail else "OK"
                    print(f"\n[{step}|PROM|{tag}] {action} → {new_loc} | {_clean_obs(new_obs)[:80]}")

                self._handle_room_change(new_loc, new_obs, action, seed + step, verbose)
                location, observation = new_loc, new_obs
                self._last_obs, self._last_score = new_obs, new_score
                history.append(("PROM", action, _clean_obs(new_obs)[:80]))
                action_done = True
                break

            if action_done:
                if self._is_done(observation):
                    break
                continue

            loc = self._get_loc(self._current_key)

            # ── 5. EXPLORATION BIAS → BFS ────────────────────────────────
            if (loc.steps_here >= self.MAX_STEPS_ROOM and has_nav
                    and self._bfs_cooldown == 0 and self._bfs_repeat < 4
                    and not location.startswith("Unknown")):
                if verbose:
                    print(f"\n[{step}|EXPLORE] {loc.steps_here} steps → BFS")
                if await self._trigger_bfs(client, verbose):
                    continue

            # ── 6. STAGNATION → BFS ──────────────────────────────────────
            if (has_nav and self.memory.is_stagnant(10)
                    and self._bfs_cooldown == 0 and self._bfs_repeat < 4
                    and not location.startswith("Unknown")):
                if verbose:
                    print(f"\n[{step}|STAGNANT] → BFS")
                if await self._trigger_bfs(client, verbose):
                    continue

            # ── 7. LLM STEP ──────────────────────────────────────────────
            prompt = self._build_prompt(observation, location, loc)
            response = call_llm(prompt, SYSTEM_PROMPT, seed=seed + step)
            thought, tool_name, tool_args = self._parse_response(response, tool_names)
            tool_name, tool_args = self._sanitize(tool_name, tool_args, tool_names, location, loc)

            if verbose:
                print(f"\n[{step}|LLM] {thought}\n  → {tool_name}({tool_args})")

            if tool_name == "navigate_to_frontier":
                if self._bfs_repeat >= 4 or self._bfs_cooldown > 0 or location.startswith("Unknown"):
                    fb = self._escape_action(location, loc)
                    new_obs, new_score, new_loc = await self._do(client, fb, location, has_get_loc)
                    moves += 1
                    self._handle_room_change(new_loc, new_obs, fb, seed + step, verbose)
                    location, observation = new_loc, new_obs
                    self._last_obs, self._last_score = new_obs, new_score
                    history.append((thought, f"fb:{fb}", _clean_obs(new_obs)[:80]))
                    continue
                if await self._trigger_bfs(client, verbose):
                    continue
                fb = self._escape_action(location, loc)
                new_obs, new_score, new_loc = await self._do(client, fb, location, has_get_loc)
                moves += 1
                self._handle_room_change(new_loc, new_obs, fb, seed + step, verbose)
                location, observation = new_loc, new_obs
                self._last_obs, self._last_score = new_obs, new_score
                history.append((thought, f"fb:{fb}", _clean_obs(new_obs)[:80]))
                continue

            # Normal tool call
            try:
                r = await client.call_tool(tool_name, tool_args)
                new_obs = self._extract(r)
            except Exception as e:
                new_obs = f"Error: {e}"

            if verbose:
                print(f"  OBS: {_clean_obs(new_obs)[:120]}")

            new_score = self._parse_score(new_obs) or self._last_score

            if tool_name == "play_action":
                action = tool_args.get("action", "look").lower()
                loc.tried.add(action)
                new_loc = await self._get_location(client, has_get_loc, new_obs)
                self.memory.update(location, new_loc, action, self._last_score, new_score)
                moves += 1

                if self._is_fail(new_obs):
                    loc.failed.add(action)
                    self._consec_fails += 1
                else:
                    self._consec_fails = 0

                self._handle_room_change(new_loc, new_obs, action, seed + step, verbose)
                location = new_loc

            self._last_obs, self._last_score = new_obs, new_score
            observation = new_obs
            self.memory.current_location = location
            history.append((thought, tool_args.get("action", tool_name)
                           if tool_name == "play_action" else tool_name,
                           _clean_obs(new_obs)[:80]))
            if self._is_done(new_obs):
                break

        return RunResult(
            final_score=self._last_score, max_score=self._max_score, moves=moves,
            locations_visited=self.memory.visited_locations,
            game_completed=self._is_done(observation), history=history,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _handle_room_change(self, new_loc, new_obs, action, seed, verbose):
        """Update room key and init if we entered a new room."""
        new_key = _room_key(new_loc, new_obs)
        if new_key != self._current_key:
            self._current_key = new_key
            if action.lower() in ALL_MOVE:
                self._last_dir_in = action
            self._consec_fails = 0
            self._init_room(new_key, new_obs, seed, verbose)

    async def _do(self, client, action, location, has_get_loc):
        try:
            r = await client.call_tool("play_action", {"action": action})
            new_obs = self._extract(r)
        except Exception as e:
            new_obs = f"Error: {e}"
        new_score = self._parse_score(new_obs) or self._last_score
        new_loc = await self._get_location(client, has_get_loc, new_obs)
        self.memory.update(location, new_loc, action, self._last_score, new_score)
        self.memory.current_location = new_loc
        return new_obs, new_score, new_loc

    async def _trigger_bfs(self, client, verbose) -> bool:
        try:
            r = await client.call_tool("navigate_to_frontier", {})
            text = self._extract(r)
            DIRS = set(CARDINAL_10 + ["enter", "exit"])
            path = [ln.strip().lower() for ln in text.strip().split("\n")
                    if ln.strip().lower() in DIRS]
            if path:
                if path == self._last_bfs_path:
                    self._bfs_repeat += 1
                else:
                    self._bfs_repeat = 0
                    self._last_bfs_path = path[:]
                self._autopilot.extend(path)
                if verbose:
                    print(f"  [BFS path: {path}]")
                return True
        except Exception:
            pass
        return False

    def _escape_action(self, location: str, loc: LocationState) -> str:
        if self._last_dir_in:
            rev = REVERSE_DIR.get(self._last_dir_in)
            if rev and rev not in loc.failed:
                return rev
        for d in CARDINAL_10 + ["enter", "exit"]:
            if d not in loc.tried and d not in loc.failed:
                return d
        exits = self.memory.location_graph.get(location, {})
        if exits:
            return list(exits.keys())[0]
        for d in CARDINAL_10:
            if d not in loc.failed:
                return d
        return "south"

    def _build_prompt(self, obs: str, location: str, loc: LocationState) -> str:
        parts = []
        clean = _clean_obs(obs)

        if loc.failed:
            parts.append(f"⛔ FORBIDDEN (failed): {', '.join(sorted(loc.failed))}")
        if loc.tried:
            parts.append(f"Tried: {', '.join(sorted(loc.tried)[:15])}")
        if loc.outcomes:
            lines = [f"  {a}: {o}" for a, o in list(loc.outcomes.items())[-5:]]
            parts.append("Outcomes:\n" + "\n".join(lines))

        # Suggest object interactions if directions exhausted
        dirs_tried = sum(1 for d in CARDINAL_10 if d in loc.tried or d in loc.failed)
        if dirs_tried >= 6:
            nouns = _extract_nouns_from_obs(clean)
            untried_interactions = []
            for noun in nouns[:5]:
                for verb in ["examine", "take", "open"]:
                    a = f"{verb} {noun}"
                    if a not in loc.tried and a not in loc.failed:
                        untried_interactions.append(a)
            if untried_interactions:
                parts.append(f"💡 TRY OBJECTS: {', '.join(untried_interactions[:5])}")

        if loc.steps_here >= self.MAX_STEPS_ROOM:
            parts.append(f"⚠️ {loc.steps_here} steps — interact with objects or LEAVE")
        if self.memory.is_stagnant(10):
            parts.append("⚠️ Score frozen 10+ steps — try examining/taking objects!")

        parts.append(f"\nLocation: {location} | Score frozen: {self.memory._no_score_steps}")
        parts.append(f"Visited: {len(self.memory.visited_locations)} locations")
        if self.memory.action_history:
            parts.append(f"Last: {' → '.join(self.memory.action_history[-5:])}")
        parts.append(f"\n=== OBSERVATION ===\n{clean}")
        parts.append("\nWhat do you do next?")
        return "\n".join(parts)

    def _parse_response(self, response: str, valid_tools: list) -> tuple:
        thought, tool_name, tool_args = "No reasoning", "play_action", {"action": "look"}
        for line in response.strip().split("\n"):
            line = line.strip()
            up = line.upper()
            if up.startswith("THOUGHT:"):
                thought = line.split(":", 1)[1].strip()
            elif up.startswith("TOOL:"):
                raw = re.sub(r'[*` ]', '', line.split(":", 1)[1].strip()).lower()
                tool_name = raw.split("(")[0] if raw else "play_action"
            elif up.startswith("ARGS:"):
                args_str = line.split(":", 1)[1].strip()
                try:
                    parsed = json.loads(args_str.replace("'", '"'))
                    if isinstance(parsed, dict):
                        tool_args = parsed
                    elif isinstance(parsed, str):
                        tool_args = {"action": parsed}
                except Exception:
                    # Try to extract action from various patterns
                    m = re.search(r'"action"\s*:\s*"([^"]+)"', args_str)
                    if m:
                        tool_args = {"action": m.group(1)}
                    else:
                        # Maybe it's just a bare string like: northeast
                        bare = args_str.strip().strip('"').strip("'").lower()
                        if bare and len(bare) < 40:
                            tool_args = {"action": bare}
                        else:
                            tool_args = {"action": "look"}
        return thought, tool_name, tool_args

    def _sanitize(self, tool_name, tool_args, valid_tools, location, loc):
        ALIASES = {
            "action": "play_action", "do": "play_action",
            "map": "get_map", "mem": "memory", "state": "memory",
            "inv": "inventory", "navigate": "navigate_to_frontier",
            "frontier": "navigate_to_frontier", "bfs": "navigate_to_frontier",
        }
        if tool_name not in valid_tools:
            tool_name = ALIASES.get(tool_name, "play_action")
        if tool_name != "play_action":
            return tool_name, tool_args

        # FIX: LLM sometimes generates {'direction': 'up', 'action': 'look'}
        # or {'key': 'northeast', 'action': 'look'} → Pydantic rejects extra keys
        # Extract the actual intended action from ANY key
        action = tool_args.get("action", "")
        if not action or action == "look":
            # Check other keys for a direction or command
            for k, v in tool_args.items():
                if k != "action" and isinstance(v, str) and v.strip():
                    val = v.strip().lower()
                    if val in ALL_MOVE or len(val) < 30:
                        action = val
                        break
        if not action:
            action = "look"

        # Strip to only "action" key (Pydantic strict)
        tool_args = {"action": action}

        action = re.sub(r'[*`]', '', action.lower().strip())
        action = " ".join(action.split())
        words = action.split()
        if len(words) >= 2 and words[0] == "go" and words[1] in ALL_MOVE:
            action = " ".join(words[1:])
            words = action.split()
        if words and words[0] in BAD_VERBS:
            words[0] = BAD_VERBS[words[0]]
            action = " ".join(words)
        tool_args["action"] = action

        if action in loc.failed:
            tool_args["action"] = self._escape_action(location, loc)

        return tool_name, tool_args

    async def _get_location(self, client, has_get_loc, obs) -> str:
        if has_get_loc:
            try:
                r = await client.call_tool("get_location", {})
                raw = self._extract(r)
                if ":" in raw:
                    loc = raw.split(":", 1)[1].strip()
                    if loc and loc not in ("Unknown", "None"):
                        return loc
            except Exception:
                pass
        # Text fallback
        clean = _clean_obs(obs)
        return self._extract_location_from_text(clean)

    def _extract_location_from_text(self, obs: str) -> str:
        MSG = [
            "you ", "there ", "that ", "it ", "ok,", "oof",
            "already", "can't", "not ", "only ", "just ",
            "have", "hear", "look ", "see ", "but ", "this ",
            "nothing", "don't", "doesn't", "won't", "grunk",
            "error",
        ]
        for line in obs.strip().split("\n"):
            line = line.strip()
            if not line or len(line) > 40:
                continue
            if not line[0].isupper():
                continue
            if line[-1] in ("!", "?", ".", ",", ";", ":"):
                continue
            if any(kw in line.lower() for kw in MSG):
                continue
            return line
        return "Unknown"

    def _extract(self, result) -> str:
        if hasattr(result, "content") and result.content:
            return result.content[0].text
        if isinstance(result, list) and result:
            item = result[0]
            return item.text if hasattr(item, "text") else str(item)
        return str(result)

    def _parse_score(self, text: str) -> Optional[int]:
        for p in [r'\[Score:\s*(\d+)', r'Score:\s*(\d+)', r'score[:\s]+(\d+)']:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _is_done(self, text: str) -> bool:
        return any(p in text.lower() for p in ["game over", "you have died", "you are dead"])


async def test_agent():
    from fastmcp import Client
    agent = StudentAgent()
    async with Client("mcp_server.py", timeout=120) as client:
        result = await agent.run(client=client, game="lostpig",
                                  max_steps=20, seed=42, verbose=True)
        print(f"\nScore: {result.final_score} | Locations: {len(result.locations_visited)}")

if __name__ == "__main__":
    asyncio.run(test_agent())
