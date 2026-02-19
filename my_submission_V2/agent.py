"""
Student Agent for Text Adventure Games - Improved Implementation v2

Improvements over v1:
1. Location tracked via MCP get_location tool (not fragile obs parsing)
2. Failure phrase detection (catches "Grunk not see that", "you can't", etc.)
3. Per-location failed action memory exposed in prompt
4. Cycle detection (4-step pattern matching)
5. Stagnation detection (no score change for N steps)
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
# LLM Configuration - DO NOT MODIFY
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


REFLEXION_PROMPT = """You are reviewing the last 5 steps of a text adventure game agent.
Based on the recent history below, write ONE sentence summarizing:
- What actions failed or had no effect
- What the agent should avoid doing next

Be specific and concise. Example: "Moving north repeatedly failed; I should try examining objects or going south instead."

Recent history:
{history}

Your one-sentence reflection:"""


def generate_reflexion(history: list[str], seed: int) -> str:
    """Ask LLM to reflect on the last 5 steps and return a one-sentence note."""
    history_str = "\n".join(f"  - {h}" for h in history[-5:])
    prompt = REFLEXION_PROMPT.format(history=history_str)
    try:
        note = call_llm(prompt, "You are a concise game strategy analyst.", seed=seed, max_tokens=80)
        # Keep only first sentence, strip newlines
        note = note.strip().split("\n")[0].split(". ")[0] + "."
        return note
    except Exception:
        return ""


@dataclass
class RunResult:
    """Result of running the agent. Do not modify this class."""
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

SYSTEM_PROMPT = """You are an expert text adventure game player. Your goal is to maximize your score and explore as many new locations as possible.

════════════════════════════════════════
⛔ ABSOLUTE RULES — READ FIRST, ALWAYS
════════════════════════════════════════
1. NEVER execute an action listed under "FORBIDDEN ACTIONS HERE" — not once, not ever.
   These actions have already been tried and have NO effect. Repeating them wastes moves.
2. If the observation contains "[No effect]" — that action is banned at this location forever.
3. If you are cycling (same location, same actions) — STOP and pick a direction you have NOT tried.
4. If score hasn't changed in 12+ steps — try something radically different (new room, new object).

════════════════════════════════════════
RESPONSE FORMAT (strict, no markdown):
════════════════════════════════════════
THOUGHT: <why this action, max 2 sentences>
TOOL: <tool_name>
ARGS: <JSON with double quotes>

════════════════════════════════════════
VALID COMMANDS for play_action:
════════════════════════════════════════
Movement : north, south, east, west, up, down, enter, exit  (or: n, s, e, w, u, d)
Objects  : take <item>, drop <item>, open <thing>, examine <thing>, read <thing>
Other    : look, inventory, turn on lamp, turn off lamp, wait, attack <x> with <y>

FORBIDDEN verbs (parser won't accept): check, inspect, search, grab, use, help, investigate, go

════════════════════════════════════════
EXPLORATION STRATEGY:
════════════════════════════════════════
- Always try unexplored directions before revisiting known rooms
- Pick up EVERYTHING you can carry (lamp, sword, keys, rope, etc.)
- Turn on the lamp BEFORE entering dark areas
- Open all containers (mailbox, chest, box, bag, etc.)
- Read any leaflet, note, sign, or inscription you find
- Examine every interesting object mentioned in the description

AVAILABLE TOOLS:
- play_action : execute a game command
- memory      : get state summary (score, failed actions, recent history)
- get_map     : see explored locations and connections
- inventory   : check what you carry
- get_location: get current location name"""


# =============================================================================
# Memory System
# =============================================================================

class AgentMemory:
    """
    Tracks game state to prevent loops and guide exploration.
    
    Key features:
    - Per-location failed action tracking (prof's main hint)
    - Cycle detection using a rolling window of (location, action) pairs
    - Location graph for navigation awareness
    - Score-change history to identify productive actions
    """

    def __init__(self):
        # Per-location: set of actions that had no effect or failed
        self.failed_actions: dict[str, set[str]] = defaultdict(set)
        # Per-location: set of actions that successfully changed state
        self.successful_actions: dict[str, set[str]] = defaultdict(set)
        # Set of visited location names
        self.visited_locations: set[str] = set()
        # Graph: location -> {direction: destination_location}
        self.location_graph: dict[str, dict[str, str]] = defaultdict(dict)
        # Rolling window of (location, action) for cycle detection
        self._recent_states: deque = deque(maxlen=12)
        # Full action history
        self.action_history: list[str] = []
        # Score history (step -> score)
        self.score_history: list[int] = [0]
        # Last known location
        self.current_location: str = "Unknown"
        # Count of steps with no score change (for stagnation detection)
        self._no_score_change_steps: int = 0
        # Reflexion: accumulated self-reflection notes (one per 5-step interval)
        self.reflexion_notes: list[str] = []

    # Phrases that indicate an action failed, even if obs text changed
    FAILURE_PHRASES = [
        "i don't understand", "i don't know the word",
        "you can't go that way", "you can't see any",
        "grunk not see that", "that's not something",
        "nothing happens", "you can't", "there is no",
        "i don't know how to", "that verb", "not open",
        "already", "you already", "doesn't work",
        "[no effect",  # injected by our mcp_server
    ]

    def update(self, location: str, action: str, obs_before: str, obs_after: str,
               score_before: int, score_after: int, direction: Optional[str] = None):
        """Update memory after an action."""
        self.visited_locations.add(location)
        self.action_history.append(action)
        self.score_history.append(score_after)
        self._recent_states.append((location, action))

        score_changed = score_after > score_before
        obs_changed = obs_before.strip() != obs_after.strip()

        # Detect failure even when obs text changes (error messages)
        obs_lower = obs_after.lower()
        is_failure_phrase = any(p in obs_lower for p in self.FAILURE_PHRASES)

        is_failed = is_failure_phrase or (not obs_changed and not score_changed)

        if is_failed:
            self.failed_actions[location].add(action.lower())
        else:
            self.successful_actions[location].add(action.lower())

        # Track score stagnation
        if not score_changed:
            self._no_score_change_steps += 1
        else:
            self._no_score_change_steps = 0

        # Update location graph if a directional move succeeded
        if direction and obs_changed and not is_failure_phrase:
            new_loc = obs_after.strip().split("\n")[0].strip()
            if new_loc and new_loc != location and len(new_loc) < 80:
                self.location_graph[location][direction] = new_loc

        self.current_location = location

    def is_action_failed(self, location: str, action: str) -> bool:
        """Check if this action is known to fail at this location."""
        return action.lower().strip() in self.failed_actions.get(location, set())

    def is_cycling(self) -> bool:
        """
        Detect if agent is stuck in a cycle (not just immediate repetition).
        Checks if the last 4 (location, action) pairs repeat the 4 before them.
        """
        if len(self._recent_states) < 8:
            return False
        states = list(self._recent_states)
        return states[-4:] == states[-8:-4]

    def is_stagnant(self, threshold: int = 6) -> bool:
        """Check if score hasn't improved in `threshold` steps."""
        return self._no_score_change_steps >= threshold

    def get_failed_here(self, location: str) -> list[str]:
        """Return list of failed actions at this location."""
        return sorted(self.failed_actions.get(location, set()))

    def get_context_string(self, location: str) -> str:
        """Build a structured context string for the prompt."""
        lines = []

        # Score and exploration stats
        lines.append(f"Score: {self.score_history[-1]} | Steps since score change: {self._no_score_change_steps}")
        lines.append(f"Locations explored: {len(self.visited_locations)}")

        # Failed actions at current location
        failed = self.get_failed_here(location)
        if failed:
            lines.append(f"Failed here (DO NOT REPEAT): {', '.join(failed)}")

        # Cycle warning
        if self.is_cycling():
            lines.append("⚠ CYCLE DETECTED: You are going in circles. Choose a completely different action.")

        if self.is_stagnant():
            lines.append("⚠ STAGNATION: Score unchanged for many steps. Try a radically different approach.")

        # Known exits from this location
        known_exits = self.location_graph.get(location, {})
        if known_exits:
            exits_str = ", ".join(f"{d}→{dest}" for d, dest in known_exits.items())
            lines.append(f"Known exits from here: {exits_str}")

        return "\n".join(lines)


# =============================================================================
# Agent
# =============================================================================

MOVEMENT_DIRECTIONS = {
    "north", "south", "east", "west", "up", "down", "enter", "exit",
    "n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw",
    "northeast", "northwest", "southeast", "southwest",
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
}


class StudentAgent:
    """
    Improved ReAct Agent with robust memory and anti-loop strategies.
    """

    def __init__(self):
        self.memory = AgentMemory()
        self.recent_tool_sequence: deque = deque(maxlen=5)
        self._last_obs: str = ""
        self._last_score: int = 0

    async def run(
        self,
        client,
        game: str,
        max_steps: int,
        seed: int,
        verbose: bool = False,
    ) -> RunResult:
        """Main ReAct loop."""
        history = []
        moves = 0

        # List available tools
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        has_get_location = "get_location" in tool_names

        # Initial observation via look
        result = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_result(result)

        # Get real location name via dedicated tool (avoids parsing obs)
        location = await self._get_location(client, has_get_location, observation)
        self.memory.visited_locations.add(location)
        self._last_obs = observation

        if verbose:
            print(f"\n=== INITIAL ===\n{observation}\n")

        for step in range(1, max_steps + 1):
            # ── Reflexion every 5 steps ──────────────────────────────────────
            if step > 1 and step % 5 == 0 and self.memory.action_history:
                note = generate_reflexion(self.memory.action_history, seed=seed + step)
                if note:
                    self.memory.reflexion_notes.append(note)
                    if verbose:
                        print(f"\n[REFLEXION step {step}] {note}")
            # ─────────────────────────────────────────────────────────────────

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

            try:
                result = await client.call_tool(tool_name, tool_args)
                new_obs = self._extract_result(result)
            except Exception as e:
                new_obs = f"Error calling {tool_name}: {e}"

            if verbose:
                print(f"OBS:     {new_obs[:250]}")

            new_score = self._parse_score(new_obs) or self._last_score

            if tool_name == "play_action":
                action = tool_args.get("action", "look")
                direction = action.lower() if action.lower() in MOVEMENT_DIRECTIONS else None

                self.memory.update(
                    location=location,
                    action=action.lower(),
                    obs_before=self._last_obs,
                    obs_after=new_obs,
                    score_before=self._last_score,
                    score_after=new_score,
                    direction=direction,
                )
                moves += 1

                # Update location via MCP tool (reliable)
                new_location = await self._get_location(client, has_get_location, new_obs)
                if new_location != location:
                    self.memory.visited_locations.add(new_location)
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
            max_score=350,
            moves=moves,
            locations_visited=self.memory.visited_locations,
            game_completed=self._is_game_over(observation),
            history=history,
        )

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    def _build_prompt(self, observation: str, memory_ctx: str) -> str:
        """Build a structured prompt with forbidden actions first for maximum attention."""
        parts = []

        # 1. FORBIDDEN ACTIONS - top of prompt, maximum salience
        location = self.memory.current_location
        failed = self.memory.get_failed_here(location)
        if failed:
            parts.append(
                f"FORBIDDEN ACTIONS AT '{location}' (DO NOT USE - zero effect guaranteed):\n"
                f"   {', '.join(failed)}\n"
                f"Any of the above = wasted move. Choose something else."
            )

        # 2. Reflexion notes (accumulated self-reflections every 5 steps)
        if self.memory.reflexion_notes:
            notes_str = "\n".join(f"  [{i+1}] {n}" for i, n in enumerate(self.memory.reflexion_notes[-3:]))
            parts.append(f"LESSONS LEARNED (from self-reflection):\n{notes_str}")

        # 3. Cycle / stagnation warnings
        if self.memory.is_cycling():
            parts.append("WARNING CYCLE: You keep repeating the same actions. Pick a completely different approach.")
        if self.memory.is_stagnant(threshold=12):
            parts.append("WARNING STAGNATION: Score frozen for 12+ steps. Change strategy radically.")

        # 4. Game state
        parts.append(f"\n=== GAME STATE ===\n{memory_ctx}")

        # 5. Recent action trail
        if self.memory.action_history:
            recent = self.memory.action_history[-5:]
            parts.append(f"Last actions: {' → '.join(recent)}")

        # 6. Current observation
        parts.append(f"\n=== CURRENT OBSERVATION ===\n{observation}")

        parts.append("\nWhat do you do next? (forbidden actions above are BANNED)")
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Response Parsing
    # -------------------------------------------------------------------------

    def _parse_response(self, response: str, valid_tools: list[str]) -> tuple[str, str, dict]:
        """Parse THOUGHT / TOOL / ARGS from LLM response."""
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
    # Tool Call Sanitization & Anti-loop
    # -------------------------------------------------------------------------

    def _sanitize_tool_call(
        self, tool_name: str, tool_args: dict, valid_tools: list[str], location: str
    ) -> tuple[str, dict]:
        """Fix tool name, fix invalid verbs, apply anti-loop overrides."""
        # Fix tool name
        TOOL_ALIASES = {
            "action": "play_action", "do": "play_action", "command": "play_action",
            "map": "get_map", "location": "get_map",
            "mem": "memory", "state": "memory", "status": "memory",
            "inv": "inventory", "items": "inventory",
        }
        if tool_name not in valid_tools:
            tool_name = TOOL_ALIASES.get(tool_name, "play_action")

        if tool_name != "play_action":
            return tool_name, tool_args

        # Fix action string
        action = tool_args.get("action", "look").lower().strip()
        action = re.sub(r'[*`]', '', action).strip()
        action = " ".join(action.split())  # normalize whitespace

        # Fix invalid leading verbs
        words = action.split()
        if words and words[0] in INVALID_VERBS:
            words[0] = INVALID_VERBS[words[0]]
            action = " ".join(words)

        tool_args["action"] = action

        # Hard block: if this action is known to fail here, force exploration
        # This is a programmatic override — LLM cannot bypass it
        if self.memory.is_action_failed(location, action):
            if verbose_override := True:
                pass  # override silently
            action = self._find_alternative_action(location)
            tool_args["action"] = action

        # Cycle override: if stuck in a cycle, force map consultation or look
        if self.memory.is_cycling():
            # Alternate between get_map and exploring different direction
            if list(self.recent_tool_sequence).count("get_map") < 1:
                return "get_map", {}
            else:
                action = self._find_alternative_action(location)
                tool_args["action"] = action

        return tool_name, tool_args

    def _find_alternative_action(self, location: str) -> str:
        """Find an action not yet tried/failed at this location."""
        failed = self.memory.failed_actions.get(location, set())
        # Try unexplored directions first
        for direction in ["exit", "down" , "north", "south", "east", "west", "up", "enter"]:
            if direction not in failed:
                return direction
        # Fallback: look around or check inventory
        for fallback in ["look", "inventory", "examine all"]:
            if fallback not in failed:
                return fallback
        return "look"

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    async def _get_location(self, client, has_get_location: bool, fallback_obs: str) -> str:
        """Get current location name reliably via MCP tool or fallback."""
        if has_get_location:
            try:
                result = await client.call_tool("get_location", {})
                raw = self._extract_result(result)
                # Format: "Current location: Outside"
                if ":" in raw:
                    return raw.split(":", 1)[1].strip()
                return raw.strip()
            except Exception:
                pass
        # Fallback: first short line of observation (imperfect but ok)
        for line in fallback_obs.strip().split("\n"):
            line = line.strip()
            if line and len(line) < 80 and not any(
                p in line.lower() for p in ["score:", "moves:", "[no effect"]
            ):
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
        """Extract score from observation text."""
        for pattern in [r'\[Score:\s*(\d+)', r'Score:\s*(\d+)', r'score[:\s]+(\d+)']:
            m = re.search(pattern, text, re.IGNORECASE)
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
        print(f"Locations ({len(result.locations_visited)}): {result.locations_visited}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
