"""
Student Agent V10 — Location-Centric Architecture

Following professor hints:
1. Function to detect new location (Jericho API via get_location tool)
2. On new location: get_valid_actions from Jericho (authoritative!)
3. Log of tried actions + LLM-summarized outcomes per room
4. Log of promising actions per room (LLM-extracted on entry)
5. LLM function that extracts promising actions from observation
6. Exploration bias: leave after MAX_STEPS_PER_ROOM steps

KEY FIX vs V9: valid_dirs from Jericho replaces blind sweep.
In the Hole/junction, Jericho returns ["east","north","southeast","southwest"] — never south.
"""

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
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "0").strip() in ("1", "true", "yes")
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
_local_pipeline = None

if USE_LOCAL_MODEL:
    import torch
    from transformers import pipeline as _hf_pipeline
    _local_pipeline = _hf_pipeline(
        "text-generation", model=LOCAL_MODEL_ID,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    LLM_CLIENT = None
else:
    _hf_token = os.getenv("HF_TOKEN")
    if not _hf_token:
        raise ValueError("HF_TOKEN not found.")
    LLM_CLIENT = InferenceClient(token=_hf_token)


def call_llm(prompt: str, system: str, seed: int, max_tokens: int = 512) -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    if USE_LOCAL_MODEL and _local_pipeline is not None:
        out = _local_pipeline(messages, max_new_tokens=max_tokens,
                              temperature=0.0001, do_sample=True)
        return out[0]["generated_text"][-1]["content"]
    resp = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL, messages=messages,
        temperature=0.0, max_tokens=max_tokens, seed=seed,
    )
    return resp.choices[0].message.content


# =============================================================================
# Per-location state (professor hints 3 & 4)
# =============================================================================

@dataclass
class LocationState:
    """All knowledge about one physical room."""
    key: str
    tried: set = field(default_factory=set)           # actions executed here
    outcomes: dict = field(default_factory=dict)      # action -> one-line summary
    promising: deque = field(default_factory=deque)   # LLM-queued actions
    valid_dirs: list = field(default_factory=list)    # from Jericho
    valid_actions: list = field(default_factory=list) # full valid list from Jericho
    steps_here: int = 0
    obs_on_entry: str = ""
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
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert text adventure game player. Maximize score and explore new locations.

RESPONSE FORMAT (strict, no extra lines):
THOUGHT: <why, 1-2 sentences>
TOOL: <tool_name>
ARGS: {"key": "value"}

VALID GAME COMMANDS for play_action:
- Directions: north, south, east, west, up, down, ne, nw, se, sw (also: enter, exit)
- Objects: take <item>, drop <item>, open <thing>, examine <thing>, read <thing>
- Other: look, inventory, wait

FORBIDDEN verbs: check, inspect, search, grab, use, help, go, investigate, get

TOOLS AVAILABLE:
- play_action          : execute game command
- get_valid_actions    : list ALL valid actions right now (USE FIRST IN NEW ROOM)
- suggest_actions      : untried dirs + BFS hint
- navigate_to_frontier : BFS autopilot to nearest unexplored room
- memory               : game state summary
- get_map              : explored map
- inventory            : items you carry
- get_location         : current room name

STRATEGY:
1. In a new room: call get_valid_actions first
2. Try ALL valid directions before spending time on objects
3. Pick up items (torch, lamp, sword, key, rope, coin, etc.)
4. Leave room if 10+ steps with no score change"""

PROMISING_EXTRACT_SYSTEM = "You output ONLY valid JSON arrays. No explanations."
PROMISING_EXTRACT_PROMPT = """Text adventure room. Extract the most promising actions to try.
Prioritize: (1) directions from valid_actions, (2) taking items, (3) opening containers, (4) examining unusual objects.
Do NOT include already-tried actions.

Already tried: {tried}
Valid actions from Jericho: {valid_actions}
Observation: {obs}

Return JSON array only, e.g.: ["east", "take torch", "open mailbox"]"""

OUTCOME_SYSTEM = "You summarize game events in one sentence."
OUTCOME_PROMPT = "In '{location}', action '{action}' produced: {obs}\nOne sentence summary:"


# =============================================================================
# Memory
# =============================================================================

class AgentMemory:
    FAILURE_PHRASES = [
        "i don't understand", "i don't know the word",
        "you can't go that way", "you can't see any",
        "that's not something", "nothing happens", "you can't",
        "there is no", "i don't know how to", "that verb",
        "not open", "doesn't work", "[no effect",
    ]

    @staticmethod
    def room_key(location: str, obs: str = "") -> str:
        """Unique key per physical room. Unknown rooms keyed by obs fingerprint."""
        if not location.startswith("Unknown") or not obs:
            return location
        return f"Unknown:{hash(obs[:80]) & 0xFFFF}"

    def __init__(self):
        self.failed_actions: dict = defaultdict(set)
        self.successful_actions: dict = defaultdict(set)
        self.visited_locations: set = set()
        self.location_graph: dict = defaultdict(dict)
        self._recent_states: deque = deque(maxlen=12)
        self.action_history: list = []
        self.score_history: list = [0]
        self.current_location: str = "Unknown"
        self._no_score_change_steps: int = 0

    def update(self, location, new_location, action, obs_before, obs_after, score_before, score_after):
        self.visited_locations.add(location)
        self.action_history.append(action)
        self.score_history.append(score_after)
        self._recent_states.append((location, action))
        score_changed = score_after > score_before
        obs_changed = obs_before.strip() != obs_after.strip()
        is_failure = any(p in obs_after.lower() for p in self.FAILURE_PHRASES)
        location_changed = new_location != location
        is_failed = is_failure or (not obs_changed and not score_changed and not location_changed)
        key = self.room_key(location, obs_after)
        if is_failed:
            self.failed_actions[key].add(action.lower())
        else:
            self.successful_actions[key].add(action.lower())
        self._no_score_change_steps = 0 if score_changed else self._no_score_change_steps + 1
        self.current_location = new_location

    def is_action_failed(self, location, action, obs=""):
        key = self.room_key(location, obs)
        return action.lower().strip() in self.failed_actions.get(key, set())

    def get_failed_here(self, location, obs=""):
        key = self.room_key(location, obs)
        return sorted(self.failed_actions.get(key, set()))

    def is_cycling(self):
        if len(self._recent_states) < 8:
            return False
        s = list(self._recent_states)
        return s[-4:] == s[-8:-4]

    def is_stagnant(self, threshold=15):
        return self._no_score_change_steps >= threshold

    def get_summary(self, location, obs=""):
        lines = [
            f"Score steps frozen: {self._no_score_change_steps}",
            f"Locations visited: {len(self.visited_locations)}",
        ]
        failed = self.get_failed_here(location, obs)
        if failed:
            lines.append(f"Failed here: {', '.join(failed)}")
        if self.is_cycling():
            lines.append("WARNING: CYCLING — move to new room")
        if self.is_stagnant():
            lines.append("WARNING: STAGNANT 15+ steps — navigate_to_frontier")
        return "\n".join(lines)


# =============================================================================
# Student Agent
# =============================================================================

MOVE_DIRS = {
    "north","south","east","west","up","down","enter","exit",
    "n","s","e","w","u","d","ne","nw","se","sw",
    "northeast","northwest","southeast","southwest",
}

BAD_VERBS = {
    "check":"examine","inspect":"examine","search":"look",
    "grab":"take","pick":"take","use":"examine",
    "investigate":"examine","get":"take","go":"north",
    "hit":"attack","kill":"attack",
}


class StudentAgent:
    MAX_STEPS_PER_ROOM = 12  # exploration bias (professor hint 6)

    def __init__(self):
        self.memory = AgentMemory()
        self._last_obs = ""
        self._last_score = 0
        self._max_score = 0
        self._autopilot_queue: deque = deque()
        self._bfs_repeat_count = 0
        self._last_bfs_path: list = []
        self._bfs_cooldown = 0
        self._navigate_fail_count = 0
        self._location_states: dict = {}
        self._current_loc_key = ""

    # -------------------------------------------------------------------------
    # Location state
    # -------------------------------------------------------------------------

    def _get_loc_state(self, location: str, obs: str) -> LocationState:
        key = AgentMemory.room_key(location, obs)
        if key not in self._location_states:
            self._location_states[key] = LocationState(key=key)
        return self._location_states[key]

    # Professor hint 5: LLM function to extract promising actions
    def _llm_extract_promising(self, location: str, obs: str, valid_actions: list,
                                tried: set, seed: int) -> list:
        try:
            prompt = PROMISING_EXTRACT_PROMPT.format(
                tried=sorted(tried)[:15],
                valid_actions=valid_actions[:20],
                obs=obs[:500],
            )
            raw = call_llm(prompt, PROMISING_EXTRACT_SYSTEM, seed=seed, max_tokens=200)
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                actions = json.loads(m.group(0))
                return [a.lower().strip() for a in actions if isinstance(a, str) and a.strip()]
        except Exception:
            pass
        # Fallback: return valid directions only
        return [a for a in valid_actions if a.lower() in MOVE_DIRS]

    # Professor hint 3: LLM outcome summary
    def _llm_outcome_summary(self, location: str, action: str, obs: str, seed: int) -> str:
        try:
            prompt = OUTCOME_PROMPT.format(location=location, action=action, obs=obs[:200])
            result = call_llm(prompt, OUTCOME_SYSTEM, seed=seed, max_tokens=50)
            return result.strip().split("\n")[0][:100]
        except Exception:
            return obs[:80]

    # Professor hint 2: initialize on room entry
    async def _on_enter_location(self, client, location: str, obs: str,
                                  seed: int, has_valid_actions: bool, verbose: bool):
        loc = self._get_loc_state(location, obs)
        if loc.initialized:
            return
        loc.initialized = True
        loc.obs_on_entry = obs

        # Get valid actions from Jericho (authoritative list)
        if has_valid_actions:
            try:
                result = await client.call_tool("get_valid_actions", {})
                text = self._extract_text(result)
                for line in text.split("\n"):
                    if "Directions:" in line:
                        parts = line.split("Directions:", 1)[1].strip()
                        loc.valid_dirs = [d.strip() for d in parts.split(",") if d.strip()]
                    elif "Actions:" in line:
                        parts = line.split("Actions:", 1)[1].strip()
                        loc.valid_actions = [a.strip() for a in parts.split(",") if a.strip()]
                if verbose:
                    print(f"[ENTRY '{location}'] dirs={loc.valid_dirs} actions={loc.valid_actions[:6]}")
            except Exception as e:
                if verbose:
                    print(f"[ENTRY] get_valid_actions error: {e}")

        all_valid = loc.valid_dirs + loc.valid_actions

        # LLM extracts promising actions (professor hint 5)
        failed_here = self.memory.get_failed_here(location, obs)
        promising = self._llm_extract_promising(location, obs, all_valid, loc.tried | set(failed_here), seed)
        for a in promising:
            if a not in loc.tried and a not in failed_here:
                loc.promising.append(a)
        if verbose:
            print(f"[ENTRY '{location}'] promising={list(loc.promising)}")

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    async def run(self, client, game: str, max_steps: int, seed: int, verbose=False) -> RunResult:
        history = []
        moves = 0

        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        has_get_loc = "get_location" in tool_names
        has_nav = "navigate_to_frontier" in tool_names
        has_valid = "get_valid_actions" in tool_names

        # Initial observation
        r = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_text(r)
        self._last_obs = observation
        location = await self._get_location(client, has_get_loc, observation)
        self.memory.visited_locations.add(location)
        self._max_score = self._parse_max_score(observation) or 350

        if verbose:
            print(f"\n=== INITIAL ===\n{observation}\n[Score: ?/{self._max_score} | Location: {location}]")

        # Initialize first room
        self._current_loc_key = AgentMemory.room_key(location, observation)
        await self._on_enter_location(client, location, observation, seed, has_valid, verbose)

        for step in range(1, max_steps + 1):
            loc = self._get_loc_state(location, observation)
            loc.steps_here += 1

            # ── 1. BFS AUTOPILOT ─────────────────────────────────────────────
            if self._autopilot_queue:
                direction = self._autopilot_queue.popleft()
                if verbose:
                    print(f"\n--- Step {step} [AUTOPILOT] ---\nDIRECTION: {direction} ({len(self._autopilot_queue)} remaining)")
                try:
                    r = await client.call_tool("play_action", {"action": direction})
                    new_obs = self._extract_text(r)
                except Exception as e:
                    new_obs = f"Error: {e}"
                    self._autopilot_queue.clear()

                new_score = self._parse_score(new_obs) or self._last_score
                new_loc = await self._get_location(client, has_get_loc, new_obs)
                self.memory.update(location, new_loc, direction,
                                   self._last_obs, new_obs, self._last_score, new_score)
                moves += 1

                if verbose:
                    print(f"OBS: {new_obs[:180]}")

                if "[No effect" in new_obs or new_loc == "Unknown":
                    self._autopilot_queue.clear()
                    self._bfs_repeat_count += 1

                if not self._autopilot_queue:
                    self._bfs_cooldown = 10
                    if new_loc != "Unknown":
                        new_key = AgentMemory.room_key(new_loc, new_obs)
                        if new_key not in self._location_states or not self._location_states[new_key].initialized:
                            await self._on_enter_location(client, new_loc, new_obs, seed + step, has_valid, verbose)
                        if verbose:
                            print(f"[BFS] Arrived at {new_loc} — cooldown {self._bfs_cooldown}")

                if new_loc != location:
                    self.memory.visited_locations.add(new_loc)
                    if new_loc != "Unknown":
                        self._navigate_fail_count = 0
                        self._bfs_repeat_count = 0
                        self._last_bfs_path = []

                location = new_loc
                self._last_obs = new_obs
                self._last_score = new_score
                observation = new_obs
                self.memory.current_location = location
                history.append(("AUTOPILOT", direction, new_obs[:80]))
                if self._is_done(new_obs):
                    break
                continue

            # ── 2. DETECT NEW LOCATION (professor hint 1) ────────────────────
            new_key = AgentMemory.room_key(location, observation)
            if new_key != self._current_loc_key:
                self._current_loc_key = new_key
                if new_key not in self._location_states or not self._location_states[new_key].initialized:
                    if verbose:
                        print(f"\n[NEW ROOM DETECTED] {location}")
                    await self._on_enter_location(client, location, observation,
                                                   seed + step, has_valid, verbose)
                    loc = self._get_loc_state(location, observation)

            # ── 3. PROMISING QUEUE (professor hints 4 & 5) ───────────────────
            while loc.promising:
                action = loc.promising.popleft()
                failed_here = self.memory.get_failed_here(location, observation)
                if action in loc.tried or action in failed_here:
                    continue
                # Execute mechanically without LLM
                if verbose:
                    print(f"\n--- Step {step} [PROMISING] ---\nACTION: {action} ({len(loc.promising)} remaining)")
                try:
                    r = await client.call_tool("play_action", {"action": action})
                    new_obs = self._extract_text(r)
                except Exception as e:
                    new_obs = f"Error: {e}"

                if verbose:
                    print(f"OBS: {new_obs[:180]}")

                loc.tried.add(action)
                new_score = self._parse_score(new_obs) or self._last_score
                new_loc = await self._get_location(client, has_get_loc, new_obs)
                self.memory.update(location, new_loc, action,
                                   self._last_obs, new_obs, self._last_score, new_score)
                moves += 1

                # Professor hint 3: log outcome with LLM summary
                loc.outcomes[action] = self._llm_outcome_summary(
                    location, action, new_obs, seed + step)

                if new_loc != location:
                    self.memory.visited_locations.add(new_loc)
                    if new_loc != "Unknown":
                        self._navigate_fail_count = 0
                        self._bfs_repeat_count = 0
                        self._last_bfs_path = []
                    new_key2 = AgentMemory.room_key(new_loc, new_obs)
                    if new_key2 not in self._location_states or not self._location_states[new_key2].initialized:
                        await self._on_enter_location(client, new_loc, new_obs,
                                                       seed + step, has_valid, verbose)

                location = new_loc
                self._last_obs = new_obs
                self._last_score = new_score
                observation = new_obs
                self.memory.current_location = location
                history.append(("PROMISING", action, new_obs[:80]))
                if self._is_done(new_obs):
                    break
                break  # one action per step

            if self._is_done(observation):
                break

            # Re-fetch loc after possible room change
            loc = self._get_loc_state(location, observation)

            # ── 4. EXPLORATION BIAS (professor hint 6) ────────────────────────
            if self._bfs_cooldown > 0:
                self._bfs_cooldown -= 1

            if (loc.steps_here >= self.MAX_STEPS_PER_ROOM and has_nav
                    and location != "Unknown" and self._bfs_cooldown == 0
                    and self._bfs_repeat_count < 2 and not self._autopilot_queue
                    and not loc.promising):
                if verbose:
                    print(f"\n[EXPLORE BIAS] {loc.steps_here} steps in '{location}' — forcing BFS")
                bfs_ok = await self._trigger_bfs(client, location, verbose)
                if bfs_ok:
                    continue

            # Stagnation safety net
            if (has_nav and location != "Unknown" and self.memory.is_stagnant(15)
                    and self._bfs_repeat_count < 2 and self._bfs_cooldown == 0
                    and not self._autopilot_queue and not loc.promising):
                bfs_ok = await self._trigger_bfs(client, location, verbose)
                if bfs_ok:
                    if verbose:
                        print(f"\n[BFS STAGNANT step {step}]")
                    continue

            # ── 5. LLM STEP (when promising is empty) ────────────────────────
            mem_summary = self.memory.get_summary(location, observation)
            prompt = self._build_prompt(observation, mem_summary, loc)
            response = call_llm(prompt, SYSTEM_PROMPT, seed=seed + step)
            thought, tool_name, tool_args = self._parse_response(response, tool_names)
            tool_name, tool_args = self._sanitize(tool_name, tool_args, tool_names, location, observation)

            if verbose:
                print(f"\n--- Step {step} ---")
                print(f"THOUGHT: {thought}")
                print(f"TOOL:    {tool_name}({tool_args})")

            # navigate_to_frontier
            if tool_name == "navigate_to_frontier":
                if location == "Unknown" or self._bfs_repeat_count >= 2 or self._bfs_cooldown > 0:
                    if verbose:
                        print("[BFS] Blocked → fallback")
                    self._navigate_fail_count += 1
                    fallback = self._fallback_action(location, loc)
                    r = await client.call_tool("play_action", {"action": fallback})
                    new_obs = self._extract_text(r)
                    new_score = self._parse_score(new_obs) or self._last_score
                    new_loc = await self._get_location(client, has_get_loc, new_obs)
                    self.memory.update(location, new_loc, fallback,
                                       self._last_obs, new_obs, self._last_score, new_score)
                    moves += 1
                    if new_loc != location:
                        self.memory.visited_locations.add(new_loc)
                    location = new_loc
                    self._last_obs = new_obs
                    self._last_score = new_score
                    observation = new_obs
                    self.memory.current_location = location
                    history.append((thought, f"nav→{fallback}", new_obs[:80]))
                    if verbose:
                        print(f"OBS: {new_obs[:180]}")
                    continue

                bfs_ok = await self._trigger_bfs(client, location, verbose)
                if not bfs_ok:
                    self._bfs_repeat_count += 1
                    fallback = self._fallback_action(location, loc)
                    r = await client.call_tool("play_action", {"action": fallback})
                    new_obs = self._extract_text(r)
                    new_score = self._parse_score(new_obs) or self._last_score
                    new_loc = await self._get_location(client, has_get_loc, new_obs)
                    self.memory.update(location, new_loc, fallback,
                                       self._last_obs, new_obs, self._last_score, new_score)
                    moves += 1
                    location = new_loc
                    self._last_score = new_score
                    observation = new_obs
                history.append((thought, "navigate_to_frontier", observation[:80]))
                self._last_obs = observation
                self.memory.current_location = location
                continue

            # Normal tool call
            try:
                r = await client.call_tool(tool_name, tool_args)
                new_obs = self._extract_text(r)
            except Exception as e:
                new_obs = f"Error {tool_name}: {e}"

            if verbose:
                print(f"OBS:     {new_obs[:200]}")

            new_score = self._parse_score(new_obs) or self._last_score
            ms = self._parse_max_score(new_obs)
            if ms:
                self._max_score = ms

            if tool_name == "play_action":
                action = tool_args.get("action", "look").lower()
                loc.tried.add(action)
                new_loc = await self._get_location(client, has_get_loc, new_obs)
                self.memory.update(location, new_loc, action,
                                   self._last_obs, new_obs, self._last_score, new_score)
                moves += 1
                if new_loc != location:
                    self.memory.visited_locations.add(new_loc)
                    if new_loc != "Unknown":
                        self._navigate_fail_count = 0
                        self._bfs_repeat_count = 0
                        self._last_bfs_path = []
                    new_key3 = AgentMemory.room_key(new_loc, new_obs)
                    if new_key3 not in self._location_states or not self._location_states[new_key3].initialized:
                        await self._on_enter_location(client, new_loc, new_obs,
                                                       seed + step, has_valid, verbose)
                location = new_loc

            self._last_obs = new_obs
            self._last_score = new_score
            observation = new_obs
            self.memory.current_location = location
            action_str = tool_args.get("action", tool_name) if tool_name == "play_action" else tool_name
            history.append((thought, action_str, new_obs[:80]))
            if self._is_done(new_obs):
                break

        return RunResult(
            final_score=self._last_score, max_score=self._max_score, moves=moves,
            locations_visited=self.memory.visited_locations,
            game_completed=self._is_done(observation), history=history,
        )

    # -------------------------------------------------------------------------
    # BFS helpers
    # -------------------------------------------------------------------------

    async def _trigger_bfs(self, client, location: str, verbose: bool) -> bool:
        try:
            r = await client.call_tool("navigate_to_frontier", {})
            text = self._extract_text(r)
            path = self._parse_path(text)
            if path:
                if path == self._last_bfs_path:
                    self._bfs_repeat_count += 1
                else:
                    self._bfs_repeat_count = 0
                    self._last_bfs_path = path[:]
                self._autopilot_queue.extend(path)
                if verbose:
                    print(f"[BFS] Path: {path}")
                return True
        except Exception:
            pass
        return False

    def _parse_path(self, text: str) -> list:
        DIRS = {"north","south","east","west","up","down","enter","exit",
                "northeast","northwest","southeast","southwest","n","s","e","w","u","d"}
        return [line.strip().lower() for line in text.strip().split("\n")
                if line.strip().lower() in DIRS]

    # -------------------------------------------------------------------------
    # Prompt building (professor hint 6: exploration bias)
    # -------------------------------------------------------------------------

    def _build_prompt(self, observation: str, mem_summary: str, loc: LocationState) -> str:
        parts = []
        location = self.memory.current_location

        # FORBIDDEN first
        failed = self.memory.get_failed_here(location, observation)
        if failed:
            parts.append(f"FORBIDDEN AT '{location}' (DO NOT USE):\n   {', '.join(failed)}")

        # Exploration bias warning
        if loc.steps_here >= self.MAX_STEPS_PER_ROOM:
            parts.append(
                f"EXPLORATION BIAS: {loc.steps_here} steps in this room. "
                "Move to another room or call navigate_to_frontier NOW."
            )

        # Cycle/stagnation
        if self.memory.is_cycling():
            parts.append("CYCLE DETECTED: Move to a different room NOW.")
        if self.memory.is_stagnant():
            parts.append("STAGNATION 15+ steps: Call navigate_to_frontier.")

        # Valid untried directions (from Jericho, authoritative)
        if loc.valid_dirs:
            untried = [d for d in loc.valid_dirs if d not in loc.tried
                       and d not in self.memory.get_failed_here(location, observation)]
            if untried:
                parts.append(f"VALID UNTRIED DIRECTIONS (from Jericho): {', '.join(untried)}")

        # Outcomes of what's been tried here
        if loc.outcomes:
            lines = [f"  {a}: {o}" for a, o in list(loc.outcomes.items())[-5:]]
            parts.append("OUTCOMES IN THIS ROOM:\n" + "\n".join(lines))

        parts.append(f"\n=== GAME STATE ===\n{mem_summary}")

        if self.memory.action_history:
            parts.append(f"Last 3 actions: {' → '.join(self.memory.action_history[-3:])}")

        parts.append(f"\n=== OBSERVATION ===\n{observation}")
        parts.append("\nWhat do you do next?")
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Fallback action using loc.valid_dirs (not guessing)
    # -------------------------------------------------------------------------

    def _fallback_action(self, location: str, loc: LocationState) -> str:
        failed = self.memory.get_failed_here(location, self._last_obs)
        tried = loc.tried

        # Use Jericho valid dirs first (most reliable)
        for d in loc.valid_dirs:
            if d not in failed and d not in tried:
                return d

        # Parse from observation text
        for d in self._parse_exits_from_obs(self._last_obs):
            if d not in failed:
                return d

        # Known graph exits
        for d in self.memory.location_graph.get(location, {}).keys():
            if d not in failed:
                return d

        # Last resort: try untried standard directions
        for d in ["east", "west", "north", "south", "up", "down",
                  "northeast", "southeast", "southwest", "northwest"]:
            if d not in failed:
                return d
        return "look"

    _EXIT_RE = re.compile(
        r'\b(north|south|east|west|up|down|northeast|northwest|southeast|southwest|enter|exit)\b',
        re.IGNORECASE
    )

    def _parse_exits_from_obs(self, obs: str) -> list:
        TRIGGERS = ["go", "way", "lead", "only", "tunnel", "door", "passage",
                    "exit", "stairs", "path", "corridor"]
        exits = []
        for line in obs.lower().split("\n"):
            if any(t in line for t in TRIGGERS):
                exits.extend(m.group(1).lower() for m in self._EXIT_RE.finditer(line))
        return list(dict.fromkeys(exits))

    # -------------------------------------------------------------------------
    # Response parsing & sanitization
    # -------------------------------------------------------------------------

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

    def _sanitize(self, tool_name: str, tool_args: dict,
                   valid_tools: list, location: str, obs: str = "") -> tuple:
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
        # "go north" → "north"
        if len(words) >= 2 and words[0] == "go" and words[1] in MOVE_DIRS:
            action = " ".join(words[1:])
            words = action.split()
        # Fix bad verbs
        if words and words[0] in BAD_VERBS:
            words[0] = BAD_VERBS[words[0]]
            action = " ".join(words)
        tool_args["action"] = action

        if self.memory.is_action_failed(location, action, self._last_obs):
            loc = self._get_loc_state(location, self._last_obs)
            tool_args["action"] = self._fallback_action(location, loc)

        return tool_name, tool_args

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    async def _get_location(self, client, has_get_loc: bool, fallback_obs: str) -> str:
        if has_get_loc:
            try:
                r = await client.call_tool("get_location", {})
                raw = self._extract_text(r)
                if ":" in raw:
                    loc = raw.split(":", 1)[1].strip()
                    if loc and loc not in ("Unknown", "None"):
                        return loc
            except Exception:
                pass
        # Text fallback
        MSG = ["you ", "there ", "that ", "it ", "ok,", "oof", "already", "can't",
               "not ", "only ", "just ", "have", "hear", "look ", "see ", "but ",
               "score:", "moves:", "nothing", "don't", "doesn't", "won't"]
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

    def _extract_text(self, result) -> str:
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

    def _parse_max_score(self, text: str) -> Optional[int]:
        m = re.search(r'Score:\s*\d+\s*/\s*(\d+)', text, re.IGNORECASE)
        return int(m.group(1)) if m else None

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
    import asyncio
    asyncio.run(test_agent())
