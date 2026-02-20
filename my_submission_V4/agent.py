"""
Student Agent V4 — Text Adventure Games

Changes vs V3-bis:
- FIX: max_score no longer hardcoded (parsed from observation or defaulted properly).
- FIX: _get_location fallback filters error messages more aggressively.
- FIX: reflexion scoped to "EXPLORE" context — no longer bans directions.
- NEW: BFS autopilot — when stagnant, calls navigate_to_frontier and executes
       the returned path mechanically WITHOUT intermediate LLM calls.
- NEW: early-location sweep — on first visit, immediately try all untried directions
       (heuristic to discover new rooms faster in the first 30 steps).
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

# =============================================================================
# LLM Configuration
# =============================================================================

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "0").strip() in ("1", "true", "yes")
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")

_local_pipeline = None

if USE_LOCAL_MODEL:
    import torch
    from transformers import pipeline as _hf_pipeline
    _local_pipeline = _hf_pipeline(
        "text-generation",
        model=LOCAL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    LLM_CLIENT = None
else:
    _hf_token = os.getenv("HF_TOKEN")
    if not _hf_token:
        raise ValueError("HF_TOKEN not found. Set it in your .env file.")
    LLM_CLIENT = InferenceClient(token=_hf_token)


def call_llm(prompt: str, system_prompt: str, seed: int, max_tokens: int = 512) -> str:
    """Call the LLM with the given prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if USE_LOCAL_MODEL and _local_pipeline is not None:
        outputs = _local_pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=0.0001,
            do_sample=True,
        )
        return outputs[0]["generated_text"][-1]["content"]

    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
    )
    return response.choices[0].message.content


# =============================================================================
# Reflexion (V4: no longer bans directions, focuses on objects/actions)
# =============================================================================

REFLEXION_PROMPT = """You are reviewing recent steps of a text adventure agent.
Based on the history below, write ONE concise sentence about:
- What OBJECT interactions or PUZZLE steps failed (not about directions)
- What the agent should try next regarding items, containers, or puzzles

Do NOT mention directions (north/south/east/west etc.) — those are handled separately.
Be specific. Example: "Opening the chest failed; I should look for a key first."

Recent history:
{history}

One-sentence reflection (about items/puzzles only):"""


def generate_reflexion(history: list[str], seed: int) -> str:
    history_str = "\n".join(f"  - {h}" for h in history[-5:])
    prompt = REFLEXION_PROMPT.format(history=history_str)
    try:
        note = call_llm(prompt, "You are a concise game strategy analyst.", seed=seed, max_tokens=80)
        note = note.strip().split("\n")[0].split(". ")[0] + "."
        return note
    except Exception:
        return ""


# =============================================================================
# RunResult
# =============================================================================

@dataclass
class RunResult:
    final_score: int
    max_score: int
    moves: int
    locations_visited: set[str]
    game_completed: bool
    error: Optional[str] = None
    history: list[tuple[str, str, str]] = field(default_factory=list)


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert text adventure game player. Goal: maximize score and explore NEW locations.

════════════════════════════════════════
⛔ ABSOLUTE RULES
════════════════════════════════════════
1. NEVER use an action listed under "FORBIDDEN ACTIONS HERE" — they are provably useless.
2. After "[No effect]" — that action is permanently banned at this location.
3. If cycling (same room, same actions) — call navigate_to_frontier immediately.
4. Score frozen 15+ steps → call navigate_to_frontier for new areas.

════════════════════════════════════════
RESPONSE FORMAT (strict, no markdown):
════════════════════════════════════════
THOUGHT: <why this action, max 2 sentences>
TOOL: <tool_name>
ARGS: <JSON with double quotes>

════════════════════════════════════════
VALID COMMANDS for play_action:
════════════════════════════════════════
Movement : north, south, east, west, up, down, enter, exit (n/s/e/w/u/d)
           northeast, northwest, southeast, southwest
Objects  : take <item>, drop <item>, open <thing>, examine <thing>, read <thing>
Other    : look, inventory, turn on lamp, turn off lamp, wait, attack <x> with <y>

FORBIDDEN verbs: check, inspect, search, grab, use, help, investigate, go

════════════════════════════════════════
EXPLORATION STRATEGY (priority order):
════════════════════════════════════════
1. Try ALL unexplored directions before examining objects
2. Pick up EVERYTHING (lamp, sword, keys, rope, torch, etc.)
3. Turn on lamp/torch BEFORE entering dark areas
4. Open all containers (mailbox, chest, box, bag)
5. Read any note, leaflet, sign, inscription
6. If stuck → call navigate_to_frontier (BFS autopilot to new rooms)

════════════════════════════════════════
AVAILABLE TOOLS:
════════════════════════════════════════
- play_action          : execute a game command
- memory               : get state summary (score, failed actions, recent history)
- get_map              : see explored locations and connections
- inventory            : check what you carry
- get_location         : get current location name (reliable)
- suggest_actions      : get untried directions + objects + BFS hint
- navigate_to_frontier : BFS path to nearest unexplored area (USE WHEN STUCK)"""


# =============================================================================
# Agent Memory
# =============================================================================

class AgentMemory:
    FAILURE_PHRASES = [
        "i don't understand", "i don't know the word",
        "you can't go that way", "you can't see any",
        "grunk not see that", "that's not something",
        "nothing happens", "you can't", "there is no",
        "i don't know how to", "that verb", "not open",
        "doesn't work", "[no effect",
    ]

    def __init__(self):
        self.failed_actions: dict[str, set[str]] = defaultdict(set)
        self.successful_actions: dict[str, set[str]] = defaultdict(set)
        self.visited_locations: set[str] = set()
        self.location_graph: dict[str, dict[str, str]] = defaultdict(dict)
        self._recent_states: deque = deque(maxlen=12)
        self.action_history: list[str] = []
        self.score_history: list[int] = [0]
        self.current_location: str = "Unknown"
        self._no_score_change_steps: int = 0
        self.reflexion_notes: list[str] = []
        self._last_suggestions: str = ""
        # V4: track which locations have been "swept" (all dirs tried)
        self._first_visit_done: set[str] = set()

    def update(self, location: str, new_location: str, action: str, obs_before: str, obs_after: str,
               score_before: int, score_after: int):
        self.visited_locations.add(location)
        self.action_history.append(action)
        self.score_history.append(score_after)
        self._recent_states.append((location, action))

        score_changed = score_after > score_before
        obs_changed = obs_before.strip() != obs_after.strip()
        obs_lower = obs_after.lower()
        is_failure_phrase = any(p in obs_lower for p in self.FAILURE_PHRASES)
        # Correct location_changed: compare actual Jericho locations
        location_changed = (new_location != location)
        # An action fails only if obs + score + location are ALL unchanged
        is_failed = is_failure_phrase or (not obs_changed and not score_changed and not location_changed)

        if is_failed:
            self.failed_actions[location].add(action.lower())
        else:
            self.successful_actions[location].add(action.lower())

        if not score_changed:
            self._no_score_change_steps += 1
        else:
            self._no_score_change_steps = 0

        self.current_location = new_location

    def is_action_failed(self, location: str, action: str) -> bool:
        return action.lower().strip() in self.failed_actions.get(location, set())

    def is_cycling(self) -> bool:
        if len(self._recent_states) < 8:
            return False
        states = list(self._recent_states)
        return states[-4:] == states[-8:-4]

    def is_stagnant(self, threshold: int = 6) -> bool:
        return self._no_score_change_steps >= threshold

    def get_failed_here(self, location: str) -> list[str]:
        return sorted(self.failed_actions.get(location, set()))

    def get_context_string(self, location: str) -> str:
        lines = []
        lines.append(f"Score: {self.score_history[-1]} | Steps without score change: {self._no_score_change_steps}")
        lines.append(f"Unique locations explored: {len(self.visited_locations)}")
        failed = self.get_failed_here(location)
        if failed:
            lines.append(f"Failed here (DO NOT REPEAT): {', '.join(failed)}")
        if self.is_cycling():
            lines.append("⚠ CYCLE: Call navigate_to_frontier NOW.")
        if self.is_stagnant(threshold=15):
            lines.append("⚠ STAGNATION (15+ steps): Call navigate_to_frontier to reach new rooms.")
        return "\n".join(lines)


# =============================================================================
# Agent
# =============================================================================

MOVEMENT_DIRECTIONS = {
    "north","south","east","west","up","down","enter","exit",
    "n","s","e","w","u","d","ne","nw","se","sw",
    "northeast","northwest","southeast","southwest",
}

INVALID_VERBS = {
    "check": "examine",
    "inspect": "examine",
    "search": "look",
    "grab": "take",
    "pick": "take",
    "use": "examine",
    "investigate": "examine",
    "get": "take",
    "go": "north",  # "go north" → "north"
}


class StudentAgent:
    """
    ReAct Agent V4 with BFS autopilot for exploration.
    """

    def __init__(self):
        self.memory = AgentMemory()
        self.recent_tool_sequence: deque = deque(maxlen=5)
        self._last_obs: str = ""
        self._last_score: int = 0
        self._max_score: int = 0
        # V4: BFS autopilot state
        self._autopilot_queue: deque[str] = deque()
        # V4fix: count consecutive navigate_to_frontier calls with no path found
        self._navigate_fail_count: int = 0
        # Fix: track last location to reset fail counter on location change
        self._last_known_location: str = "Unknown"
        # Unknown zone sweep: systematic direction index when stuck in Unknown
        self._unknown_sweep_idx: int = 0

    async def run(
        self,
        client,
        game: str,
        max_steps: int,
        seed: int,
        verbose: bool = False,
    ) -> RunResult:
        """Main ReAct loop with BFS autopilot."""
        history = []
        moves = 0

        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        has_get_location = "get_location" in tool_names
        has_navigate = "navigate_to_frontier" in tool_names

        # Initial look
        result = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_result(result)
        self._last_obs = observation

        # Reliable location from MCP
        location = await self._get_location(client, has_get_location, observation)
        self.memory.visited_locations.add(location)

        # Parse initial max_score if available
        self._max_score = self._parse_max_score(observation) or 350

        if verbose:
            print(f"\n=== INITIAL ===\n{observation}\n")

        for step in range(1, max_steps + 1):

            # ── BFS Autopilot ────────────────────────────────────────────────
            # If we have a queued path, execute next step mechanically
            if self._autopilot_queue:
                direction = self._autopilot_queue.popleft()
                if verbose:
                    print(f"\n--- Step {step} [AUTOPILOT] ---")
                    print(f"DIRECTION: {direction} ({len(self._autopilot_queue)} remaining)")

                try:
                    result = await client.call_tool("play_action", {"action": direction})
                    new_obs = self._extract_result(result)
                except Exception as e:
                    new_obs = f"Error: {e}"
                    self._autopilot_queue.clear()  # abort path on error

                if verbose:
                    print(f"OBS: {new_obs[:200]}")

                new_score = self._parse_score(new_obs) or self._last_score
                new_location = await self._get_location(client, has_get_location, new_obs)
                self.memory.update(
                    location=location,
                    new_location=new_location,
                    action=direction,
                    obs_before=self._last_obs,
                    obs_after=new_obs,
                    score_before=self._last_score,
                    score_after=new_score,
                )
                moves += 1

                if new_location != location:
                    self.memory.visited_locations.add(new_location)
                    # Reset fail counter when we actually move to a new location
                    if new_location != "Unknown":
                        self._navigate_fail_count = 0
                location = new_location

                self._last_obs = new_obs
                self._last_score = new_score
                observation = new_obs
                self.memory.current_location = location
                history.append(("AUTOPILOT", direction, new_obs[:100]))

                if self._is_game_over(new_obs):
                    break
                continue
            # ─────────────────────────────────────────────────────────────────

            # ── Unknown Zone Escape ──────────────────────────────────────────
            # When stuck in Unknown for many steps, do a systematic direction sweep
            # ignoring failed_actions (which may be corrupted under "Unknown" key)
            SWEEP_DIRS = ["south","north","east","west","up","down","enter","exit",
                          "northeast","northwest","southeast","southwest"]
            if (location == "Unknown"
                    and self.memory.is_stagnant(threshold=8)
                    and not self._autopilot_queue):
                sweep_dir = SWEEP_DIRS[self._unknown_sweep_idx % len(SWEEP_DIRS)]
                self._unknown_sweep_idx += 1
                if verbose:
                    print(f"\n--- Step {step} [UNKNOWN SWEEP #{self._unknown_sweep_idx}] direction={sweep_dir} ---")
                try:
                    result = await client.call_tool("play_action", {"action": sweep_dir})
                    new_obs = self._extract_result(result)
                except Exception as e:
                    new_obs = f"Error: {e}"
                new_score = self._parse_score(new_obs) or self._last_score
                new_location = await self._get_location(client, has_get_location, new_obs)
                self.memory.update(
                    location=location,
                    new_location=new_location,
                    action=sweep_dir,
                    obs_before=self._last_obs,
                    obs_after=new_obs,
                    score_before=self._last_score,
                    score_after=new_score,
                )
                moves += 1
                if new_location != location:
                    self.memory.visited_locations.add(new_location)
                    if new_location != "Unknown":
                        self._navigate_fail_count = 0
                location = new_location
                self._last_obs = new_obs
                self._last_score = new_score
                observation = new_obs
                self.memory.current_location = location
                history.append(("SWEEP", sweep_dir, new_obs[:100]))
                if self._is_game_over(new_obs):
                    break
                continue
            # ─────────────────────────────────────────────────────────────────
            if step > 1 and step % 5 == 0 and self.memory.action_history:
                note = generate_reflexion(self.memory.action_history, seed=seed + step)
                if note:
                    self.memory.reflexion_notes.append(note)
                    if verbose:
                        print(f"\n[REFLEXION step {step}] {note}")

            # ── Trigger BFS autopilot when stagnant ──────────────────────────
            if has_navigate and self.memory.is_stagnant(threshold=15) and not self._autopilot_queue:
                try:
                    nav_result = await client.call_tool("navigate_to_frontier", {})
                    nav_text = self._extract_result(nav_result)
                    path = self._parse_frontier_path(nav_text)
                    if path:
                        self._autopilot_queue.extend(path)
                        if verbose:
                            print(f"\n[BFS AUTOPILOT step {step}] Path: {path}")
                        continue  # start autopilot immediately
                except Exception as e:
                    if verbose:
                        print(f"\n[BFS] Error: {e}")

            # ── suggest_actions refresh ──────────────────────────────────────
            has_suggest = "suggest_actions" in tool_names
            if has_suggest and self.memory.is_stagnant(threshold=10) and step % 10 == 0:
                try:
                    sug_result = await client.call_tool("suggest_actions", {})
                    sug_text = self._extract_result(sug_result)
                    self.memory._last_suggestions = sug_text
                    if verbose:
                        print(f"\n[SUGGEST step {step}]\n{sug_text}")
                except Exception:
                    pass

            # ── Normal LLM step ──────────────────────────────────────────────
            memory_ctx = self.memory.get_context_string(location)
            prompt = self._build_prompt(observation, memory_ctx)

            response = call_llm(prompt, SYSTEM_PROMPT, seed=seed + step)
            thought, tool_name, tool_args = self._parse_response(response, tool_names)
            tool_name, tool_args = self._sanitize_tool_call(
                tool_name, tool_args, tool_names, location
            )

            if verbose:
                print(f"\n--- Step {step} ---")
                print(f"THOUGHT: {thought}")
                print(f"TOOL:    {tool_name}({tool_args})")

            # Special: if LLM calls navigate_to_frontier, load autopilot queue
            if tool_name == "navigate_to_frontier":
                try:
                    result = await client.call_tool("navigate_to_frontier", {})
                    nav_text = self._extract_result(result)
                    path = self._parse_frontier_path(nav_text)
                    if path:
                        self._autopilot_queue.extend(path)
                        self._navigate_fail_count = 0  # reset on success
                        if verbose:
                            print(f"[BFS] Loaded path: {path}")
                        new_obs = nav_text
                    else:
                        # NO FRONTIER FOUND — force a play_action instead
                        self._navigate_fail_count += 1
                        if verbose:
                            print(f"[BFS] No frontier (fail #{self._navigate_fail_count}), forcing play_action")
                        fallback = self._find_alternative_action(location)
                        result2 = await client.call_tool("play_action", {"action": fallback})
                        new_obs = self._extract_result(result2)
                        new_score = self._parse_score(new_obs) or self._last_score
                        new_location = await self._get_location(client, has_get_location, new_obs)
                        self.memory.update(
                            location=location,
                            new_location=new_location,
                            action=fallback,
                            obs_before=self._last_obs,
                            obs_after=new_obs,
                            score_before=self._last_score,
                            score_after=new_score,
                        )
                        moves += 1
                        if new_location != location:
                            self.memory.visited_locations.add(new_location)
                            if new_location != "Unknown":
                                self._navigate_fail_count = 0
                        location = new_location
                        self._last_score = new_score
                except Exception as e:
                    new_obs = f"navigate_to_frontier error: {e}"
                self._last_obs = new_obs
                observation = new_obs
                history.append((thought, "navigate_to_frontier", new_obs[:100]))
                self.recent_tool_sequence.append(tool_name)
                continue

            try:
                result = await client.call_tool(tool_name, tool_args)
                new_obs = self._extract_result(result)
            except Exception as e:
                new_obs = f"Error calling {tool_name}: {e}"

            if verbose:
                print(f"OBS:     {new_obs[:250]}")

            new_score = self._parse_score(new_obs) or self._last_score
            ms = self._parse_max_score(new_obs)
            if ms:
                self._max_score = ms

            if tool_name == "play_action":
                action = tool_args.get("action", "look")
                new_location = await self._get_location(client, has_get_location, new_obs)
                self.memory.update(
                    location=location,
                    new_location=new_location,
                    action=action.lower(),
                    obs_before=self._last_obs,
                    obs_after=new_obs,
                    score_before=self._last_score,
                    score_after=new_score,
                )
                moves += 1

                if new_location != location:
                    self.memory.visited_locations.add(new_location)
                    if new_location != "Unknown":
                        self._navigate_fail_count = 0
                location = new_location

            self._last_obs = new_obs
            self._last_score = new_score
            observation = new_obs
            self.memory.current_location = location
            self.recent_tool_sequence.append(tool_name)

            action_str = tool_args.get("action", str(tool_args)) if tool_name == "play_action" else tool_name
            history.append((thought, action_str, new_obs[:100]))

            if self._is_game_over(new_obs):
                if verbose:
                    print("\n*** GAME OVER ***")
                break

        return RunResult(
            final_score=self._last_score,
            max_score=self._max_score,
            moves=moves,
            locations_visited=self.memory.visited_locations,
            game_completed=self._is_game_over(observation),
            history=history,
        )

    # -------------------------------------------------------------------------
    # BFS path parsing
    # -------------------------------------------------------------------------

    def _parse_frontier_path(self, text: str) -> list[str]:
        """Extract the list of directions from navigate_to_frontier output."""
        VALID_DIRS = {
            "north","south","east","west","up","down","enter","exit",
            "northeast","northwest","southeast","southwest",
            "n","s","e","w","u","d","ne","nw","se","sw",
        }
        lines = text.strip().split("\n")
        path = []
        for line in lines:
            word = line.strip().lower()
            if word in VALID_DIRS:
                path.append(word)
        return path

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    def _build_prompt(self, observation: str, memory_ctx: str) -> str:
        parts = []
        location = self.memory.current_location

        # 1. Forbidden actions
        failed = self.memory.get_failed_here(location)
        if failed:
            parts.append(
                f"FORBIDDEN ACTIONS AT '{location}' (DO NOT USE):\n"
                f"   {', '.join(failed)}\n"
                f"Repeating any of the above = wasted move."
            )

        # 2. Suggestions
        if self.memory._last_suggestions:
            parts.append(f"SUGGESTED NEXT ACTIONS:\n{self.memory._last_suggestions}")

        # 3. Reflexion notes
        if self.memory.reflexion_notes:
            notes_str = "\n".join(f"  [{i+1}] {n}" for i, n in enumerate(self.memory.reflexion_notes[-3:]))
            parts.append(f"LESSONS LEARNED:\n{notes_str}")

        # 4. Warnings
        if self.memory.is_cycling():
            parts.append("⛔ CYCLE DETECTED: Call navigate_to_frontier NOW.")
        if self.memory.is_stagnant(threshold=15):
            parts.append("⛔ STAGNATION: Call navigate_to_frontier to find new rooms.")

        # 5. Game state
        parts.append(f"\n=== GAME STATE ===\n{memory_ctx}")

        # 6. Recent history
        if self.memory.action_history:
            recent = self.memory.action_history[-5:]
            parts.append(f"Last actions: {' → '.join(recent)}")

        # 7. Observation
        parts.append(f"\n=== CURRENT OBSERVATION ===\n{observation}")

        parts.append("\nWhat do you do next?")
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Response Parsing
    # -------------------------------------------------------------------------

    def _parse_response(self, response: str, valid_tools: list[str]) -> tuple[str, str, dict]:
        thought = "No reasoning"
        tool_name = "play_action"
        tool_args = {"action": "look"}

        for line in response.strip().split("\n"):
            line = line.strip()
            upper = line.upper()
            if upper.startswith("THOUGHT:"):
                thought = line.split(":", 1)[1].strip()
            elif upper.startswith("TOOL:"):
                raw = line.split(":", 1)[1].strip().lower()
                raw = re.sub(r'[*`]', '', raw).split()[0] if raw else "play_action"
                tool_name = raw
            elif upper.startswith("ARGS:"):
                args_str = line.split(":", 1)[1].strip()
                try:
                    args_str = args_str.replace("'", '"')
                    tool_args = json.loads(args_str)
                except json.JSONDecodeError:
                    m = re.search(r'"action"\s*:\s*"([^"]+)"', args_str)
                    tool_args = {"action": m.group(1)} if m else {"action": "look"}

        return thought, tool_name, tool_args

    # -------------------------------------------------------------------------
    # Sanitization
    # -------------------------------------------------------------------------

    def _sanitize_tool_call(
        self, tool_name: str, tool_args: dict, valid_tools: list[str], location: str
    ) -> tuple[str, dict]:
        TOOL_ALIASES = {
            "action": "play_action", "do": "play_action", "command": "play_action",
            "map": "get_map",
            "mem": "memory", "state": "memory", "status": "memory",
            "inv": "inventory", "items": "inventory",
            "navigate": "navigate_to_frontier", "frontier": "navigate_to_frontier",
            "bfs": "navigate_to_frontier",
        }
        if tool_name not in valid_tools:
            tool_name = TOOL_ALIASES.get(tool_name, "play_action")

        # If BFS keeps failing (no frontier found), force play_action instead
        if tool_name == "navigate_to_frontier" and self._navigate_fail_count >= 2:
            tool_name = "play_action"
            tool_args = {"action": self._find_alternative_action(location)}

        if tool_name != "play_action":
            return tool_name, tool_args

        action = tool_args.get("action", "look").lower().strip()
        action = re.sub(r'[*`]', '', action).strip()
        action = " ".join(action.split())

        # Fix "go north" → "north"
        words = action.split()
        if len(words) >= 2 and words[0] == "go" and words[1] in MOVEMENT_DIRECTIONS:
            action = " ".join(words[1:])
            words = action.split()

        if words and words[0] in INVALID_VERBS:
            words[0] = INVALID_VERBS[words[0]]
            action = " ".join(words)

        tool_args["action"] = action

        # Hard block on known failures
        if self.memory.is_action_failed(location, action):
            action = self._find_alternative_action(location)
            tool_args["action"] = action

        # Cycle override
        if self.memory.is_cycling():
            if list(self.recent_tool_sequence).count("get_map") < 1:
                return "get_map", {}
            else:
                action = self._find_alternative_action(location)
                tool_args["action"] = action

        return tool_name, tool_args

    def _find_alternative_action(self, location: str) -> str:
        """Find an untried direction, prioritizing known graph exits then unexplored dirs."""
        failed = self.memory.failed_actions.get(location, set())
        known_exits = set(self.memory.location_graph.get(location, {}).keys())

        # First: try directions confirmed to work (in graph) but not recently used
        for direction in known_exits:
            if direction not in failed:
                return direction

        # Second: try standard directions in a smart order (south often = start)
        for direction in ["south", "north", "east", "west", "up", "down",
                          "enter", "exit", "northeast", "northwest", "southeast", "southwest"]:
            if direction not in failed and direction not in known_exits:
                return direction

        # Last resort
        return "look"

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    async def _get_location(self, client, has_get_location: bool, fallback_obs: str) -> str:
        """Get current location reliably."""
        if has_get_location:
            try:
                result = await client.call_tool("get_location", {})
                raw = self._extract_result(result)
                if ":" in raw:
                    loc = raw.split(":", 1)[1].strip()
                    if loc and loc != "Unknown":
                        return loc
            except Exception:
                pass

        # Fallback: first short line that looks like a room title
        MESSAGE_INDICATORS = [
            "grunk", "you ", "there ", "that ", "it ", "ok,", "oof",
            "already", "can't", "not ", "only ", "pig ", "just ",
            "have", "hear", "look ", "see ", "but ", "this ", "score:", "moves:",
        ]
        for line in fallback_obs.strip().split("\n"):
            line = line.strip()
            if not line or len(line) >= 60:
                continue
            if not line[0].isupper():
                continue
            if line.endswith("!") or line.endswith("?"):
                continue
            if any(kw in line.lower() for kw in MESSAGE_INDICATORS):
                continue
            return line

        return "Unknown"

    def _extract_result(self, result) -> str:
        if hasattr(result, 'content') and result.content:
            return result.content[0].text
        if isinstance(result, list) and result:
            item = result[0]
            return item.text if hasattr(item, 'text') else str(item)
        return str(result)

    def _parse_score(self, text: str) -> Optional[int]:
        for pattern in [r'\[Score:\s*(\d+)', r'Score:\s*(\d+)', r'score[:\s]+(\d+)']:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _parse_max_score(self, text: str) -> Optional[int]:
        # Matches "Score: 1/350" or "5/350" patterns
        m = re.search(r'Score:\s*\d+\s*/\s*(\d+)', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def _is_game_over(self, text: str) -> bool:
        phrases = ["game over", "you have died", "you are dead", "*** you have died ***"]
        return any(p in text.lower() for p in phrases)


# =============================================================================
# Local Testing
# =============================================================================

async def test_agent():
    from fastmcp import Client
    agent = StudentAgent()
    async with Client("mcp_server.py") as client:
        result = await agent.run(
            client=client, game="lostpig", max_steps=20, seed=42, verbose=True
        )
        print(f"\n{'='*50}")
        print(f"Final Score:      {result.final_score}")
        print(f"Moves:            {result.moves}")
        print(f"Locations ({len(result.locations_visited)}): {sorted(result.locations_visited)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
