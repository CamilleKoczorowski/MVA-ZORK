"""
Student Agent V11b — Hybrid: Jericho valid actions + mechanical fallback

Architecture:
- On new room: try get_valid_actions (Jericho) with 30s timeout
  → If works: queue ONLY valid directions (0 wasted steps!)
  → If fails: queue all 10 cardinal directions (mechanical fallback)
- LLM extracts object interactions (take, open, examine) on room entry
- Strict per-room failed action tracking (obs-hash for Unknown rooms)
- BFS to frontier when stuck
- LLM called only when mechanical queue is empty

Fix vs V10: no concurrent.futures threading (was killing GIL).
Just asyncio.wait_for with generous timeout + flag to disable if it hangs.
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
ALL_MOVE_WORDS = set(CARDINAL_10 + ["enter", "exit", "n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw"])

FAILURE_PHRASES = [
    "i don't understand", "i don't know the word",
    "you can't go that way", "you can't see any",
    "that's not something", "nothing happens", "you can't",
    "there is no", "i don't know how to", "that verb",
    "not open", "doesn't work", "[no effect",
    "doesn't seem to", "isn't something", "don't need to",
    "already", "huh?", "what?", "[failed",
]

BAD_VERBS = {
    "check": "examine", "inspect": "examine", "search": "look",
    "grab": "take", "pick": "take", "use": "examine",
    "investigate": "examine", "get": "take", "go": "north",
    "hit": "attack", "kill": "attack",
}


# =============================================================================
# Per-location state
# =============================================================================

@dataclass
class LocationState:
    key: str
    tried: set = field(default_factory=set)
    outcomes: dict = field(default_factory=dict)
    promising: deque = field(default_factory=deque)
    valid_dirs: list = field(default_factory=list)      # from Jericho (if available)
    valid_actions: list = field(default_factory=list)    # non-direction actions from Jericho
    steps_here: int = 0
    initialized: bool = False
    jericho_queried: bool = False  # whether we got valid actions from Jericho


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
    @staticmethod
    def room_key(location: str, obs: str = "") -> str:
        if not location.startswith("Unknown") or not obs:
            return location
        return f"Unknown:{hash(obs[:80]) & 0xFFFF}"

    def __init__(self):
        self.failed_actions: dict = defaultdict(set)
        self.successful_actions: dict = defaultdict(set)
        self.visited_locations: set = set()
        self.location_graph: dict = defaultdict(dict)
        self._recent_pairs: deque = deque(maxlen=12)
        self.action_history: list = []
        self.score_history: list = [0]
        self.current_location: str = "Unknown"
        self._no_score_steps: int = 0

    def update(self, loc_before, loc_after, action, obs_before, obs_after,
               score_before, score_after):
        self.visited_locations.add(loc_before)
        self.visited_locations.add(loc_after)
        self.action_history.append(action)
        self.score_history.append(score_after)
        self._recent_pairs.append((loc_before, action))

        obs_lower = obs_after.lower()
        is_failure = any(p in obs_lower for p in FAILURE_PHRASES)
        obs_changed = obs_before.strip() != obs_after.strip()
        score_changed = score_after > score_before
        loc_changed = loc_after != loc_before

        key = self.room_key(loc_before, obs_before)
        if is_failure or (not obs_changed and not score_changed and not loc_changed):
            self.failed_actions[key].add(action.lower())
        else:
            self.successful_actions[key].add(action.lower())

        self._no_score_steps = 0 if score_changed else self._no_score_steps + 1
        self.current_location = loc_after

    def is_failed(self, location, action, obs=""):
        key = self.room_key(location, obs)
        return action.lower().strip() in self.failed_actions.get(key, set())

    def get_failed_here(self, location, obs=""):
        key = self.room_key(location, obs)
        return sorted(self.failed_actions.get(key, set()))

    def is_cycling(self):
        if len(self._recent_pairs) < 8:
            return False
        s = list(self._recent_pairs)
        return s[-4:] == s[-8:-4]

    def is_stagnant(self, threshold=15):
        return self._no_score_steps >= threshold

    def get_summary(self, location, obs=""):
        lines = [
            f"Score frozen: {self._no_score_steps} steps",
            f"Locations visited: {len(self.visited_locations)}",
        ]
        failed = self.get_failed_here(location, obs)
        if failed:
            lines.append(f"Failed here: {', '.join(failed)}")
        return "\n".join(lines)


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert text adventure game player. Maximize score and explore new rooms.

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
- suggest_actions: see untried directions + BFS hint
- navigate_to_frontier: BFS autopilot to nearest unexplored room
- memory: game state summary
- get_map: explored map
- inventory: check items
- get_location: current room name

RULES:
1. NEVER repeat an action listed in FORBIDDEN AT (those already failed here)
2. Try directions first, then interact with objects
3. Pick up useful items (torch, lamp, key, coin, sword, rope)
4. If stuck: call navigate_to_frontier
5. Explore as many rooms as possible"""

PROMISING_SYSTEM = "Output ONLY a JSON array. No explanation."
PROMISING_PROMPT = """Text adventure room. Extract promising actions to try.
Priority: (1) directions not yet tried, (2) take items, (3) open/examine objects.
Do NOT include already-tried actions.

Already tried: {tried}
Valid actions from game engine: {valid}
Observation: {obs}

Return JSON array only, e.g.: ["east", "take torch", "open mailbox"]"""


# =============================================================================
# Student Agent
# =============================================================================

class StudentAgent:
    MAX_STEPS_ROOM = 10
    JERICHO_TIMEOUT = 30.0  # generous timeout for get_valid_actions

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
        self._jericho_ok: bool = True  # disabled if get_valid_actions hangs

    # ── Location state management ────────────────────────────────────────

    def _get_loc(self, location: str, obs: str) -> LocationState:
        key = AgentMemory.room_key(location, obs)
        if key not in self._loc_states:
            self._loc_states[key] = LocationState(key=key)
        return self._loc_states[key]

    async def _init_room(self, client, location: str, obs: str, seed: int,
                          has_valid: bool, verbose: bool):
        """Initialize a new room: get valid actions + queue promising."""
        loc = self._get_loc(location, obs)
        if loc.initialized:
            return
        loc.initialized = True

        failed = set(self.memory.get_failed_here(location, obs))

        # 1. Try Jericho get_valid_actions (with asyncio timeout)
        got_jericho = False
        if has_valid and self._jericho_ok:
            try:
                result = await asyncio.wait_for(
                    client.call_tool("get_valid_actions", {}),
                    timeout=self.JERICHO_TIMEOUT
                )
                text = self._extract(result)
                for line in text.split("\n"):
                    if "Directions:" in line:
                        parts = line.split("Directions:", 1)[1].strip()
                        loc.valid_dirs = [d.strip() for d in parts.split(",") if d.strip()]
                    elif "Actions:" in line:
                        parts = line.split("Actions:", 1)[1].strip()
                        loc.valid_actions = [a.strip() for a in parts.split(",") if a.strip()]
                loc.jericho_queried = True
                got_jericho = bool(loc.valid_dirs or loc.valid_actions)
                if verbose:
                    print(f"  [JERICHO] dirs={loc.valid_dirs} actions={loc.valid_actions[:5]}")
            except asyncio.TimeoutError:
                self._jericho_ok = False
                if verbose:
                    print(f"  [JERICHO] TIMEOUT {self.JERICHO_TIMEOUT}s — disabling for rest of run")
            except Exception as e:
                if verbose:
                    print(f"  [JERICHO] error: {e}")

        # 2. Queue directions to try
        if got_jericho and loc.valid_dirs:
            # Only queue confirmed valid directions (huge efficiency!)
            for d in loc.valid_dirs:
                d_low = d.lower()
                if d_low not in loc.tried and d_low not in failed:
                    loc.promising.append(d_low)
        else:
            # Fallback: queue all cardinal directions
            for d in CARDINAL_10:
                if d not in loc.tried and d not in failed:
                    loc.promising.append(d)

        # 3. Queue valid non-direction actions from Jericho
        if loc.valid_actions:
            for a in loc.valid_actions[:10]:
                a_low = a.lower()
                if a_low not in loc.tried and a_low not in failed and a_low not in set(loc.promising):
                    loc.promising.append(a_low)

        # 4. LLM extracts extra promising actions (items, interactions)
        all_valid = loc.valid_dirs + loc.valid_actions
        llm_actions = self._llm_extract(location, obs, loc.tried | failed, all_valid, seed)
        for a in llm_actions:
            if a not in loc.tried and a not in failed and a not in set(loc.promising):
                loc.promising.append(a)

        if verbose:
            print(f"  [INIT '{location}'] promising={list(loc.promising)[:10]}...")

    def _llm_extract(self, location: str, obs: str, already_tried: set,
                     valid: list, seed: int) -> list:
        """Ask LLM for promising non-direction actions."""
        try:
            prompt = PROMISING_PROMPT.format(
                tried=sorted(already_tried)[:20],
                valid=valid[:20] if valid else "(unknown)",
                obs=obs[:600],
            )
            raw = call_llm(prompt, PROMISING_SYSTEM, seed=seed, max_tokens=200)
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                actions = json.loads(m.group(0))
                return [a.lower().strip() for a in actions
                        if isinstance(a, str) and a.strip() and len(a) < 40]
        except Exception:
            pass
        return []

    # ── Main run loop ────────────────────────────────────────────────────

    async def run(self, client, game: str, max_steps: int, seed: int,
                  verbose=False) -> RunResult:
        history = []
        moves = 0

        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        has_get_loc = "get_location" in tool_names
        has_nav = "navigate_to_frontier" in tool_names
        has_valid = "get_valid_actions" in tool_names

        # Initial observation
        r = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract(r)
        self._last_obs = observation
        location = await self._get_location(client, has_get_loc, observation)
        self.memory.visited_locations.add(location)

        if verbose:
            print(f"\n=== START ===\n{observation}\n[Location: {location}]")

        # Init first room
        self._current_key = AgentMemory.room_key(location, observation)
        await self._init_room(client, location, observation, seed, has_valid, verbose)

        for step in range(1, max_steps + 1):
            loc = self._get_loc(location, observation)
            loc.steps_here += 1

            if self._bfs_cooldown > 0:
                self._bfs_cooldown -= 1

            # ── 1. AUTOPILOT (BFS path) ─────────────────────────────────
            if self._autopilot:
                action = self._autopilot.popleft()
                new_obs, new_score, new_loc = await self._exec_action(
                    client, action, location, observation, has_get_loc)
                moves += 1

                if verbose:
                    print(f"\n[{step}|AUTO] {action} → {new_loc} | {new_obs[:100]}")

                if "[FAILED" in new_obs or "can't go" in new_obs.lower():
                    self._autopilot.clear()
                    self._bfs_repeat += 1

                if not self._autopilot:
                    self._bfs_cooldown = 8

                if new_loc != location:
                    await self._enter_new(client, new_loc, new_obs, seed + step, has_valid, verbose)

                location, observation = new_loc, new_obs
                self._last_obs, self._last_score = new_obs, new_score
                history.append(("AUTO", action, new_obs[:80]))
                if self._is_done(new_obs):
                    break
                continue

            # ── 2. DETECT NEW ROOM ───────────────────────────────────────
            new_key = AgentMemory.room_key(location, observation)
            if new_key != self._current_key:
                self._current_key = new_key
                await self._init_room(client, location, observation, seed + step, has_valid, verbose)
                loc = self._get_loc(location, observation)

            # ── 3. PROMISING QUEUE (mechanical, no LLM) ──────────────────
            action_done = False
            while loc.promising:
                action = loc.promising.popleft()
                failed_here = set(self.memory.get_failed_here(location, observation))
                if action in loc.tried or action in failed_here:
                    continue

                new_obs, new_score, new_loc = await self._exec_action(
                    client, action, location, observation, has_get_loc)
                moves += 1
                loc.tried.add(action)
                loc.outcomes[action] = new_obs.strip().split("\n")[0][:80]

                if verbose:
                    print(f"\n[{step}|PROM] {action} → {new_loc} | {new_obs[:100]}")

                if new_loc != location:
                    await self._enter_new(client, new_loc, new_obs, seed + step, has_valid, verbose)

                location, observation = new_loc, new_obs
                self._last_obs, self._last_score = new_obs, new_score
                history.append(("PROMISING", action, new_obs[:80]))
                action_done = True
                break

            if action_done:
                if self._is_done(observation):
                    break
                continue

            # Re-fetch loc after possible room change
            loc = self._get_loc(location, observation)

            # ── 4. EXPLORATION BIAS → BFS ────────────────────────────────
            if (loc.steps_here >= self.MAX_STEPS_ROOM and has_nav
                    and not location.startswith("Unknown")
                    and self._bfs_cooldown == 0 and self._bfs_repeat < 3):
                if verbose:
                    print(f"\n[{step}|EXPLORE] {loc.steps_here} steps → BFS")
                if await self._trigger_bfs(client, verbose):
                    continue

            # ── 5. STAGNATION → BFS ──────────────────────────────────────
            if (has_nav and self.memory.is_stagnant(12)
                    and not location.startswith("Unknown")
                    and self._bfs_cooldown == 0 and self._bfs_repeat < 3):
                if verbose:
                    print(f"\n[{step}|STAGNANT] → BFS")
                if await self._trigger_bfs(client, verbose):
                    continue

            # ── 6. LLM STEP ──────────────────────────────────────────────
            prompt = self._build_prompt(observation, location, loc)
            response = call_llm(prompt, SYSTEM_PROMPT, seed=seed + step)
            thought, tool_name, tool_args = self._parse_response(response, tool_names)
            tool_name, tool_args = self._sanitize(tool_name, tool_args, tool_names, location, observation)

            if verbose:
                print(f"\n[{step}|LLM] {thought}\n  → {tool_name}({tool_args})")

            # Handle navigate_to_frontier
            if tool_name == "navigate_to_frontier":
                if (location.startswith("Unknown") or self._bfs_repeat >= 3
                        or self._bfs_cooldown > 0):
                    fallback = self._fallback(location, loc)
                    new_obs, new_score, new_loc = await self._exec_action(
                        client, fallback, location, observation, has_get_loc)
                    moves += 1
                    if new_loc != location:
                        await self._enter_new(client, new_loc, new_obs, seed + step, has_valid, verbose)
                    location, observation = new_loc, new_obs
                    self._last_obs, self._last_score = new_obs, new_score
                    history.append((thought, f"fb:{fallback}", new_obs[:80]))
                    if verbose:
                        print(f"  [BFS blocked → {fallback}] {new_obs[:100]}")
                    continue
                if await self._trigger_bfs(client, verbose):
                    history.append((thought, "bfs", ""))
                    continue
                fallback = self._fallback(location, loc)
                new_obs, new_score, new_loc = await self._exec_action(
                    client, fallback, location, observation, has_get_loc)
                moves += 1
                if new_loc != location:
                    await self._enter_new(client, new_loc, new_obs, seed + step, has_valid, verbose)
                location, observation = new_loc, new_obs
                self._last_obs, self._last_score = new_obs, new_score
                history.append((thought, f"fb:{fallback}", new_obs[:80]))
                continue

            # Normal tool call
            try:
                r = await client.call_tool(tool_name, tool_args)
                new_obs = self._extract(r)
            except Exception as e:
                new_obs = f"Error: {e}"

            if verbose:
                print(f"  OBS: {new_obs[:150]}")

            new_score = self._parse_score(new_obs) or self._last_score

            if tool_name == "play_action":
                action = tool_args.get("action", "look").lower()
                loc.tried.add(action)
                new_loc = await self._get_location(client, has_get_loc, new_obs)
                self.memory.update(location, new_loc, action,
                                   self._last_obs, new_obs, self._last_score, new_score)
                moves += 1
                if new_loc != location:
                    await self._enter_new(client, new_loc, new_obs, seed + step, has_valid, verbose)
                location = new_loc

            self._last_obs, self._last_score = new_obs, new_score
            observation = new_obs
            self.memory.current_location = location
            history.append((thought, tool_args.get("action", tool_name)
                           if tool_name == "play_action" else tool_name, new_obs[:80]))
            if self._is_done(new_obs):
                break

        return RunResult(
            final_score=self._last_score, max_score=self._max_score, moves=moves,
            locations_visited=self.memory.visited_locations,
            game_completed=self._is_done(observation), history=history,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _enter_new(self, client, new_loc, new_obs, seed, has_valid, verbose):
        self.memory.visited_locations.add(new_loc)
        if not new_loc.startswith("Unknown"):
            self._bfs_repeat = 0
        await self._init_room(client, new_loc, new_obs, seed, has_valid, verbose)

    async def _exec_action(self, client, action, location, obs_before, has_get_loc):
        try:
            r = await client.call_tool("play_action", {"action": action})
            new_obs = self._extract(r)
        except Exception as e:
            new_obs = f"Error: {e}"

        new_score = self._parse_score(new_obs) or self._last_score
        new_loc = await self._get_location(client, has_get_loc, new_obs)
        self.memory.update(location, new_loc, action,
                           obs_before, new_obs, self._last_score, new_score)
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

    def _fallback(self, location: str, loc: LocationState) -> str:
        failed = set(self.memory.get_failed_here(location, self._last_obs))
        tried = loc.tried

        # Prefer Jericho valid dirs if available
        if loc.valid_dirs:
            for d in loc.valid_dirs:
                d_low = d.lower()
                if d_low not in failed and d_low not in tried:
                    return d_low

        for d in CARDINAL_10 + ["enter", "exit"]:
            if d not in failed and d not in tried:
                return d
        exits = self.memory.location_graph.get(location, {})
        if exits:
            return list(exits.keys())[0]
        return "look"

    def _build_prompt(self, obs: str, location: str, loc: LocationState) -> str:
        parts = []

        failed = self.memory.get_failed_here(location, obs)
        if failed:
            parts.append(f"⛔ FORBIDDEN AT '{location}' (DO NOT USE):\n   {', '.join(failed)}")

        if loc.tried:
            parts.append(f"Already tried here: {', '.join(sorted(loc.tried)[:15])}")

        if loc.outcomes:
            lines = [f"  {a}: {o}" for a, o in list(loc.outcomes.items())[-5:]]
            parts.append("Outcomes:\n" + "\n".join(lines))

        # Show valid untried directions (from Jericho if available)
        if loc.valid_dirs:
            untried = [d for d in loc.valid_dirs if d.lower() not in loc.tried
                       and d.lower() not in set(failed)]
            if untried:
                parts.append(f"VALID UNTRIED DIRECTIONS: {', '.join(untried)}")

        if loc.steps_here >= self.MAX_STEPS_ROOM:
            parts.append(f"⚠️ {loc.steps_here} steps here — LEAVE NOW")

        if self.memory.is_stagnant(12):
            parts.append("⚠️ Score frozen 12+ steps — try something new!")

        summary = self.memory.get_summary(location, obs)
        parts.append(f"\n=== STATE ===\n{summary}")

        if self.memory.action_history:
            parts.append(f"Last actions: {' → '.join(self.memory.action_history[-5:])}")

        parts.append(f"\n=== OBSERVATION ===\n{obs}")
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
                    tool_args = json.loads(args_str.replace("'", '"'))
                except Exception:
                    m = re.search(r'"action"\s*:\s*"([^"]+)"', args_str)
                    tool_args = {"action": m.group(1)} if m else {"action": "look"}
        return thought, tool_name, tool_args

    def _sanitize(self, tool_name, tool_args, valid_tools, location, obs):
        ALIASES = {
            "action": "play_action", "do": "play_action",
            "map": "get_map", "mem": "memory", "state": "memory",
            "inv": "inventory", "navigate": "navigate_to_frontier",
            "frontier": "navigate_to_frontier", "bfs": "navigate_to_frontier",
            "valid": "get_valid_actions",
        }
        if tool_name not in valid_tools:
            tool_name = ALIASES.get(tool_name, "play_action")
        if tool_name != "play_action":
            return tool_name, tool_args

        action = re.sub(r'[*`]', '', tool_args.get("action", "look").lower().strip())
        action = " ".join(action.split())
        words = action.split()
        if len(words) >= 2 and words[0] == "go" and words[1] in ALL_MOVE_WORDS:
            action = " ".join(words[1:])
            words = action.split()
        if words and words[0] in BAD_VERBS:
            words[0] = BAD_VERBS[words[0]]
            action = " ".join(words)
        tool_args["action"] = action

        if self.memory.is_failed(location, action, self._last_obs):
            loc = self._get_loc(location, self._last_obs)
            tool_args["action"] = self._fallback(location, loc)

        return tool_name, tool_args

    # ── Utility ──────────────────────────────────────────────────────────

    async def _get_location(self, client, has_get_loc, fallback_obs) -> str:
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
        MSG = ["you ", "there ", "that ", "it ", "ok,", "oof", "already", "can't",
               "not ", "only ", "just ", "have", "hear", "look ", "see ", "but ",
               "score:", "moves:", "nothing", "don't", "doesn't", "won't", "grunk"]
        for line in fallback_obs.strip().split("\n"):
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


# =============================================================================
# Test
# =============================================================================

async def test_agent():
    from fastmcp import Client
    agent = StudentAgent()
    async with Client("mcp_server.py", timeout=120) as client:
        result = await agent.run(client=client, game="lostpig",
                                  max_steps=20, seed=42, verbose=True)
        print(f"\nScore: {result.final_score} | Locations: {len(result.locations_visited)}")

if __name__ == "__main__":
    asyncio.run(test_agent())
