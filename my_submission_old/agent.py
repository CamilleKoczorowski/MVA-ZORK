"""
Student Agent for Text Adventure Games - Improved Implementation

Improvements over the baseline (example_submission/agent.py):
1. Per-location failed action memory (key insight from prof)
2. Cycle detection (not just immediate repetition)
3. Location graph for navigation awareness
4. Richer prompt with structured context
5. Score-change detection to identify productive actions
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

#LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # ou 3B pour les tests

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


def call_llm(prompt: str, system_prompt: str, seed: int, max_tokens: int = 350) -> str:
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

AVAILABLE TOOLS:
- play_action: Execute a game command
- memory: Get current state summary (score, moves, recent history)
- get_map: See explored locations and connections
- inventory: Check what you're carrying

VALID COMMANDS for play_action:
- Movement: north, south, east, west, up, down, enter, exit  (abbreviate: n, s, e, w, u, d)
- Objects: take <item>, drop <item>, open <thing>, close <thing>, examine <thing>
- Combat: attack <enemy> with <weapon>
- Other: look, inventory, read <thing>, turn on lamp, turn off lamp, wait

FORBIDDEN verbs (won't work): check, inspect, search, grab, use, help, investigate

RESPONSE FORMAT (strict, no markdown):
THOUGHT: <brief reasoning>
TOOL: <tool_name>
ARGS: <JSON>

EXPLORATION STRATEGY:
- Prioritize unexplored directions over revisiting known areas
- Pick up EVERYTHING you can carry (lamp, sword, keys, etc.)
- Turn on the lamp before entering dark areas
- Open all containers (mailbox, chest, box, etc.)
- Read any papers, notes, or signs you find
- If stuck in a location, try: examine all objects, open everything, try all directions

ANTI-LOOP RULES (critical):
- NEVER repeat an action listed as "Failed here" for the current location
- If you've tried all directions, go back and explore a different branch
- If score hasn't changed in 10+ steps, try a completely different approach"""


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

    def update(self, location: str, action: str, obs_before: str, obs_after: str, 
               score_before: int, score_after: int, direction: Optional[str] = None):
        """Update memory after an action."""
        self.visited_locations.add(location)
        self.action_history.append(action)
        self.score_history.append(score_after)
        self._recent_states.append((location, action))

        # Determine if action had any effect
        obs_changed = obs_before.strip() != obs_after.strip()
        score_changed = score_after > score_before

        if score_changed or obs_changed:
            self.successful_actions[location].add(action)
        else:
            # Action had no effect: mark as failed for this location
            self.failed_actions[location].add(action)

        # Track score stagnation
        if not score_changed:
            self._no_score_change_steps += 1
        else:
            self._no_score_change_steps = 0

        # Update location graph if a movement action succeeded
        if direction and obs_changed:
            new_location = obs_after.split("\n")[0].strip()
            if new_location != location:
                self.location_graph[location][direction] = new_location

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

    def is_stagnant(self, threshold: int = 12) -> bool:
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

        # Initial observation
        result = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_result(result)
        location = self._extract_location(observation)
        self.memory.visited_locations.add(location)
        self._last_obs = observation

        if verbose:
            print(f"\n=== INITIAL ===\n{observation}\n")

        for step in range(1, max_steps + 1):
            # Get memory summary (lightweight: no extra MCP call unless needed)
            memory_ctx = self.memory.get_context_string(location)

            # Build prompt
            prompt = self._build_prompt(observation, memory_ctx)

            # LLM call
            response = call_llm(prompt, SYSTEM_PROMPT, seed=seed + step)

            # Parse response
            thought, tool_name, tool_args = self._parse_response(response, tool_names)

            # Fix invalid verbs and apply anti-loop overrides
            tool_name, tool_args = self._sanitize_tool_call(
                tool_name, tool_args, tool_names, location
            )

            if verbose:
                print(f"\n--- Step {step} ---")
                print(f"THOUGHT: {thought}")
                print(f"TOOL:    {tool_name}({tool_args})")

            # Execute
            try:
                result = await client.call_tool(tool_name, tool_args)
                new_obs = self._extract_result(result)
            except Exception as e:
                new_obs = f"Error calling {tool_name}: {e}"

            if verbose:
                print(f"OBS:     {new_obs[:200]}")

            # Parse new score from observation
            new_score = self._parse_score(new_obs) or self._last_score

            # Update memory (only for play_action)
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

            # Update state
            self._last_obs = new_obs
            self._last_score = new_score
            observation = new_obs
            location = self._extract_location(observation)
            self.memory.current_location = location
            self.memory.visited_locations.add(location)
            self.recent_tool_sequence.append(tool_name)

            # Record
            action_str = tool_args.get("action", str(tool_args)) if tool_name == "play_action" else tool_name
            history.append((thought, action_str, new_obs[:100]))

            # Check game over
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
        """Build a structured prompt with context and anti-loop guidance."""
        parts = []

        # Memory context (score, failed actions, cycle warnings)
        parts.append(f"=== GAME STATE ===\n{memory_ctx}")

        # Recent action history (last 5)
        if self.memory.action_history:
            recent = self.memory.action_history[-5:]
            parts.append(f"\nLast actions: {' → '.join(recent)}")

        # Current observation
        parts.append(f"\n=== CURRENT OBSERVATION ===\n{observation}")

        parts.append("\nWhat do you do next?")
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

        # Anti-loop: if this action is known to fail here, force exploration
        if self.memory.is_action_failed(location, action):
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
        for direction in ["north", "south", "east", "west", "up", "down", "enter", "exit"]:
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

    def _extract_result(self, result) -> str:
        if hasattr(result, 'content') and result.content:
            return result.content[0].text
        if isinstance(result, list) and result:
            item = result[0]
            return item.text if hasattr(item, 'text') else str(item)
        return str(result)

    def _extract_location(self, observation: str) -> str:
        """Extract location name from first non-empty line of observation."""
        for line in observation.strip().split("\n"):
            line = line.strip()
            if line and len(line) < 80:  # location names are short
                return line
        return "Unknown"

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
