"""
Student Agent V11 — Mechanical-First Location-Centric Explorer

Architecture:
- On new room: queue ALL untried cardinal directions as mechanical actions
- Execute promising queue WITHOUT LLM (saves time for exploration)
- LLM only called when mechanical options exhausted
- Strict per-room failed action tracking (obs-hash for Unknown rooms)
- BFS to frontier when stuck
- No Jericho get_valid_actions (was causing hangs)

Goal: 40+ locations, score 5/7 on lostpig in 100 steps
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
    outcomes: dict = field(default_factory=dict)  # action → first line of obs
    promising: deque = field(default_factory=deque)
    steps_here: int = 0
    initialized: bool = False


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
PROMISING_PROMPT = """Text adventure room observation. Extract promising actions.
Priority: (1) untried directions, (2) take items, (3) open/examine objects.
Do NOT include already-tried actions.

Already tried: {tried}
Observation: {obs}

Return JSON array only, e.g.: ["east", "take torch", "open mailbox"]"""


# =============================================================================
# Student Agent
# =============================================================================

class StudentAgent:
    MAX_STEPS_ROOM = 10  # leave after this many steps in same room

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

    # ── Location state management ────────────────────────────────────────

    def _get_loc(self, location: str, obs: str) -> LocationState:
        key = AgentMemory.room_key(location, obs)
        if key not in self._loc_states:
            self._loc_states[key] = LocationState(key=key)
        return self._loc_states[key]

    def _init_room(self, location: str, obs: str, seed: int):
        """Initialize a new room: queue directions + LLM-extracted actions."""
        loc = self._get_loc(location, obs)
        if loc.initialized:
            return
        loc.initialized = True

        failed = set(self.memory.get_failed_here(location, obs))

        # 1. Queue all untried cardinal directions (mechanical, no LLM)
        for d in CARDINAL_10:
            if d not in loc.tried and d not in failed:
                loc.promising.append(d)

        # 2. Parse obvious actions from observation text
        obs_actions = self._parse_actions_from_obs(obs)
        for a in obs_actions:
            if a not in loc.tried and a not in failed and a not in loc.promising:
                loc.promising.append(a)

        # 3. LLM extracts extra promising actions (items, objects)
        llm_actions = self._llm_extract(location, obs, loc.tried | failed, seed)
        for a in llm_actions:
            if a not in loc.tried and a not in failed and a not in set(loc.promising):
                loc.promising.append(a)

    def _parse_actions_from_obs(self, obs: str) -> list:
        """Extract obvious interactive actions from observation text."""
        actions = []
        obs_lower = obs.lower()

        # Items to take
        TAKE_PATTERNS = [
            r'\b(?:a|an|the|some)\s+([\w\s]+?)\s+(?:is|lies|sits|rests|leans)\b',
            r'\bthere\s+(?:is|are)\s+(?:a|an|the|some)\s+([\w\s]+?)(?:\s+here|\s*[.])',
        ]
        for pat in TAKE_PATTERNS:
            for m in re.finditer(pat, obs_lower):
                item = m.group(1).strip()
                if item and len(item) < 25 and len(item.split()) <= 3:
                    actions.append(f"take {item}")

        # Directions mentioned in text
        EXIT_TRIGGERS = ["go", "way", "lead", "tunnel", "door", "passage",
                         "exit", "stairs", "path", "corridor", "opening"]
        for line in obs_lower.split("\n"):
            if any(t in line for t in EXIT_TRIGGERS):
                for m in re.finditer(
                    r'\b(north|south|east|west|up|down|northeast|northwest|southeast|southwest|enter|exit)\b',
                    line
                ):
                    d = m.group(1)
                    if d not in actions:
                        actions.append(d)

        return actions

    def _llm_extract(self, location: str, obs: str, already_tried: set, seed: int) -> list:
        """Ask LLM for promising actions (items, objects, interactions)."""
        try:
            prompt = PROMISING_PROMPT.format(
                tried=sorted(already_tried)[:20],
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
        self._init_room(location, observation, seed)

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
                    self._enter_new(new_loc, new_obs, seed + step)

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
                self._init_room(location, observation, seed + step)
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
                    self._enter_new(new_loc, new_obs, seed + step)

                location, observation = new_loc, new_obs
                self._last_obs, self._last_score = new_obs, new_score
                history.append(("PROMISING", action, new_obs[:80]))
                action_done = True
                break

            if action_done:
                if self._is_done(observation):
                    break
                continue

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

            # Handle navigate_to_frontier from LLM
            if tool_name == "navigate_to_frontier":
                if (location.startswith("Unknown") or self._bfs_repeat >= 3
                        or self._bfs_cooldown > 0):
                    # Blocked → fallback to untried direction
                    fallback = self._fallback(location, loc)
                    new_obs, new_score, new_loc = await self._exec_action(
                        client, fallback, location, observation, has_get_loc)
                    moves += 1
                    if new_loc != location:
                        self._enter_new(new_loc, new_obs, seed + step)
                    location, observation = new_loc, new_obs
                    self._last_obs, self._last_score = new_obs, new_score
                    history.append((thought, f"fb:{fallback}", new_obs[:80]))
                    if verbose:
                        print(f"  [BFS blocked → fallback: {fallback}] {new_obs[:100]}")
                    continue
                if await self._trigger_bfs(client, verbose):
                    history.append((thought, "navigate_to_frontier", ""))
                    continue
                # BFS failed → fallback
                fallback = self._fallback(location, loc)
                new_obs, new_score, new_loc = await self._exec_action(
                    client, fallback, location, observation, has_get_loc)
                moves += 1
                if new_loc != location:
                    self._enter_new(new_loc, new_obs, seed + step)
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
                    self._enter_new(new_loc, new_obs, seed + step)
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

    def _enter_new(self, new_loc, new_obs, seed):
        """Record new location and init room."""
        self.memory.visited_locations.add(new_loc)
        if not new_loc.startswith("Unknown"):
            self._bfs_repeat = 0
        self._init_room(new_loc, new_obs, seed)

    async def _exec_action(self, client, action, location, obs_before, has_get_loc):
        """Execute action and return (new_obs, new_score, new_location)."""
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
        """Pick an untried direction."""
        failed = set(self.memory.get_failed_here(location, self._last_obs))
        tried = loc.tried
        for d in CARDINAL_10 + ["enter", "exit"]:
            if d not in failed and d not in tried:
                return d
        # All tried → pick least-tried confirmed exit from graph
        exits = self.memory.location_graph.get(location, {})
        if exits:
            return list(exits.keys())[0]
        return "look"

    def _build_prompt(self, obs: str, location: str, loc: LocationState) -> str:
        parts = []

        # FORBIDDEN actions first (professor: context management is key)
        failed = self.memory.get_failed_here(location, obs)
        if failed:
            parts.append(f"⛔ FORBIDDEN AT '{location}' (DO NOT USE):\n   {', '.join(failed)}")

        # What we already tried here
        if loc.tried:
            parts.append(f"Already tried here: {', '.join(sorted(loc.tried)[:15])}")

        # Outcomes
        if loc.outcomes:
            lines = [f"  {a}: {o}" for a, o in list(loc.outcomes.items())[-5:]]
            parts.append("Outcomes:\n" + "\n".join(lines))

        # Exploration bias
        if loc.steps_here >= self.MAX_STEPS_ROOM:
            parts.append(f"⚠️ {loc.steps_here} steps here — LEAVE NOW (navigate_to_frontier or try a direction)")

        # Stagnation
        if self.memory.is_stagnant(12):
            parts.append("⚠️ Score frozen 12+ steps — try something new!")

        # Game state
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

        # Block failed actions → fallback
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
        # Text fallback
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
