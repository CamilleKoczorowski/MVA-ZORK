"""
MCP ReAct Agent for Text Adventure Games

A production-ready ReAct agent that uses FastMCP Client to play text adventures via MCP tools.
This agent connects to the Text Adventure MCP server and uses the LLM to reason and act.

Features:
- FastMCP Client integration for MCP server communication
- ReAct loop (Thought -> Tool -> Observation)
- Loop detection and action validation
- History tracking and memory management
- Score tracking and game over detection
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


@dataclass
class MCPAgentConfig:
    """Configuration for the MCP ReAct agent."""
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    game: str = "zork1"  # Default game to play
    temperature: float = 0.7
    max_tokens: int = 300
    max_history: int = 10
    verbose: bool = True


SYSTEM_PROMPT = """You are an expert text adventure game player. Your goal is to explore, collect treasures, and maximize your score.

AVAILABLE TOOLS (use these via MCP):
1. play_action - Execute game commands (north, take lamp, open mailbox, etc.)
2. memory - Get current game state, score, and recent history
3. get_map - See explored locations and connections
4. inventory - Check what you're carrying
5. hint - Get a hint if stuck
6. list_games - See available games
7. reset_game - Switch to a different game

VALID GAME COMMANDS for play_action:
- Movement: north, south, east, west, up, down, enter, exit
- Objects: take <item>, drop <item>, open <thing>, close <thing>, examine <thing>
- Light: turn on lamp, turn off lamp
- Combat: attack <enemy> with <weapon>
- Other: inventory, look, read <thing>, wait

FORBIDDEN (will NOT work): check, inspect, search, grab, use, help

RESPOND IN THIS EXACT FORMAT (no markdown):
THOUGHT: <brief reasoning about what to do next>
TOOL: <tool_name>
ARGS: <JSON arguments>

Examples:
THOUGHT: I need to see what's around me.
TOOL: play_action
ARGS: {"action": "look"}

THOUGHT: Let me check my current state and score.
TOOL: memory
ARGS: {}

THOUGHT: The mailbox might contain something useful.
TOOL: play_action
ARGS: {"action": "open mailbox"}

STRATEGY:
1. Start by looking around and checking memory
2. Explore systematically - try all directions
3. Pick up useful items (lamp, sword, etc.)
4. Open containers (mailbox, window, etc.)
5. Use get_map to avoid getting lost
6. Turn on lamp before dark areas!

DO NOT repeat the same action multiple times in a row."""


class MCPReActAgent:
    """
    A ReAct agent that plays text adventure games using MCP tools via FastMCP Client.
    
    This is the robust/production version with:
    - Full MCP integration
    - Loop detection
    - Action validation
    - Score tracking
    """
    
    def __init__(self, mcp_server_path: str, config: MCPAgentConfig = None):
        """
        Initialize the MCP ReAct agent.
        
        Args:
            mcp_server_path: Path to the MCP server script
            config: Agent configuration
        """
        load_dotenv()
        
        self.mcp_server_path = mcp_server_path
        self.config = config or MCPAgentConfig()
        
        # Override model from environment if set
        env_model = os.getenv("HF_MODEL")
        if env_model:
            self.config.model = env_model
        
        # Initialize LLM client
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Set it in your .env file.")
        self.llm = InferenceClient(token=token)
        
        # Agent state
        self.history: list[dict] = []
        self.thoughts: list[str] = []
        self.score: int = 0
        self.max_score: int = 350
        self.recent_actions: list[str] = []  # For loop detection
    
    async def run(self, max_steps: int = 100) -> dict:
        """
        Run the ReAct agent loop.
        
        Args:
            max_steps: Maximum number of steps to run
            
        Returns:
            Dictionary with game results
        """
        import time
        start_time = time.time()
        step = 0
        game_over = False
        game_name = self.config.game
        
        print("=" * 60)
        print(f"MCP ReAct Agent - Playing {game_name.upper()}")
        print(f"Model: {self.config.model}")
        print("=" * 60)
        
        # Set game as environment variable for the server
        env = os.environ.copy()
        env["GAME"] = game_name
        
        # Create transport with environment variables
        transport = StdioTransport(
            command=sys.executable,
            args=[self.mcp_server_path],
            env=env,
        )
        
        # Connect to MCP server with game environment
        async with Client(transport) as client:
            # List available tools
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            print(f"\nConnected to MCP server. Tools: {tool_names}")
            
            # Get initial observation
            result = await client.call_tool("play_action", {"action": "look"})
            observation = self._extract_result(result)
            print(f"\n{observation}\n")
            
            # Parse initial score
            self._update_score(observation)
            
            # Main ReAct loop
            for step in range(1, max_steps + 1):
                print(f"\n{'─' * 50}")
                print(f"Step {step}/{max_steps} | Score: {self.score}")
                print("─" * 50)
                
                # Build prompt with context
                prompt = self._build_prompt(observation)
                
                # Call LLM for reasoning
                response = self._call_llm(prompt)
                
                # Parse response
                thought, tool_name, tool_args = self._parse_response(response, tool_names)
                
                self.thoughts.append(thought)
                
                if self.config.verbose:
                    print(f"\n[THOUGHT] {thought}")
                    print(f"[TOOL] {tool_name}({tool_args})")
                
                # Validate and fix common issues
                tool_name, tool_args = self._validate_tool_call(tool_name, tool_args, tool_names)
                
                # Check for loops
                if tool_name == "play_action":
                    action = tool_args.get("action", "look")
                    self.recent_actions.append(action)
                    if len(self.recent_actions) > 5:
                        self.recent_actions = self.recent_actions[-5:]
                    
                    # Detect loops
                    if len(self.recent_actions) >= 3 and len(set(self.recent_actions[-3:])) == 1:
                        print(f"\n[WARNING] Loop detected - repeating '{action}'")
                        # Force a different action
                        tool_args = {"action": "look"}
                        self.recent_actions.append("look")
                
                # Execute tool via MCP
                try:
                    result = await client.call_tool(tool_name, tool_args)
                    observation = self._extract_result(result)
                    print(f"\n{observation}")
                except Exception as e:
                    observation = f"Error executing tool: {e}"
                    print(f"\n[ERROR] {e}")
                
                # Update history
                self.history.append({
                    "step": step,
                    "thought": thought,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": observation[:200]
                })
                if len(self.history) > self.config.max_history:
                    self.history = self.history[-self.config.max_history:]
                
                # Update score
                self._update_score(observation)
                
                # Check for game over
                if self._is_game_over(observation):
                    game_over = True
                    print("\n" + "=" * 60)
                    print("GAME OVER!")
                    break
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        return self._print_summary(step, elapsed_time, game_over)
    
    def _build_prompt(self, observation: str) -> str:
        """Build the prompt for the LLM with context."""
        parts = []
        
        # Score info
        parts.append(f"Current Score: {self.score}/{self.max_score}")
        
        # Recent history (compact)
        if self.history:
            parts.append("\nRecent actions:")
            for entry in self.history[-3:]:
                action = entry.get("args", {}).get("action", entry["tool"])
                result_short = entry["result"][:80] + "..." if len(entry["result"]) > 80 else entry["result"]
                parts.append(f"  > {action} -> {result_short}")
            
            # Warn about repeated actions
            if self.recent_actions and len(set(self.recent_actions[-3:])) == 1:
                parts.append(f"\n[WARNING: You've been doing '{self.recent_actions[-1]}' repeatedly. TRY SOMETHING DIFFERENT!]")
        
        # Current observation
        parts.append(f"\nCurrent situation:\n{observation}")
        parts.append("\nWhat do you do next?")
        
        return "\n".join(parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM for reasoning."""
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM Error] {e}")
            return "THOUGHT: LLM error, trying look.\nTOOL: play_action\nARGS: {\"action\": \"look\"}"
    
    def _parse_response(self, response: str, valid_tools: list[str]) -> tuple[str, str, dict]:
        """Parse the LLM response to extract thought, tool, and arguments."""
        thought = "No reasoning provided"
        tool_name = "play_action"
        tool_args = {"action": "look"}
        
        lines = response.strip().split("\n")
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            line_upper = line_clean.upper()
            
            if line_upper.startswith("THOUGHT:"):
                thought = line_clean.split(":", 1)[1].strip()
            
            elif line_upper.startswith("TOOL:"):
                raw_tool = line_clean.split(":", 1)[1].strip().lower()
                # Clean up common issues
                raw_tool = raw_tool.replace("**", "").replace("*", "").replace("`", "")
                raw_tool = raw_tool.split()[0] if raw_tool else "play_action"
                tool_name = raw_tool
            
            elif line_upper.startswith("ARGS:"):
                args_part = line_clean.split(":", 1)[1].strip()
                try:
                    # Handle various JSON formats
                    args_part = args_part.replace("'", '"')
                    tool_args = json.loads(args_part)
                except json.JSONDecodeError:
                    # Try to extract action from text
                    match = re.search(r'"action"\s*:\s*"([^"]+)"', args_part)
                    if match:
                        tool_args = {"action": match.group(1)}
                    else:
                        # Fallback: try to use the whole thing as action
                        tool_args = {"action": "look"}
        
        return thought, tool_name, tool_args
    
    def _validate_tool_call(self, tool_name: str, tool_args: dict, valid_tools: list[str]) -> tuple[str, dict]:
        """Validate and fix common tool call issues."""
        # Fix tool name
        if tool_name not in valid_tools:
            # Try common alternatives
            if tool_name in ["action", "do", "command"]:
                tool_name = "play_action"
            elif tool_name in ["map", "location"]:
                tool_name = "get_map"
            elif tool_name in ["mem", "state", "status"]:
                tool_name = "memory"
            elif tool_name in ["inv", "items"]:
                tool_name = "inventory"
            else:
                tool_name = "play_action"
        
        # Fix action in args
        if tool_name == "play_action":
            action = tool_args.get("action", "look")
            
            # Fix invalid verbs
            invalid_verb_map = {
                "check": "examine",
                "inspect": "examine",
                "search": "look",
                "grab": "take",
                "pick": "take",
                "use": "examine",
                "investigate": "examine",
            }
            
            words = action.lower().split()
            if words and words[0] in invalid_verb_map:
                words[0] = invalid_verb_map[words[0]]
                action = " ".join(words)
            
            # Clean up action
            action = action.lower().strip()
            action = action.replace("**", "").replace("*", "").replace("`", "")
            action = " ".join(action.split())
            
            tool_args["action"] = action
        
        return tool_name, tool_args
    
    def _extract_result(self, result) -> str:
        """Extract text from MCP tool result."""
        if hasattr(result, 'content') and result.content:
            return result.content[0].text
        return str(result)
    
    def _update_score(self, text: str) -> None:
        """Update score from game text."""
        # Look for score patterns
        patterns = [
            r'\+(\d+) points',
            r'Score:\s*(\d+)',
            r'Total:\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if "+" in pattern:
                    self.score += score
                else:
                    self.score = max(self.score, score)
    
    def _is_game_over(self, text: str) -> bool:
        """Check if the game is over."""
        game_over_phrases = [
            "game over",
            "you have died",
            "you are dead",
            "*** you have died ***",
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in game_over_phrases)
    
    def _print_summary(self, step: int, elapsed_time: float, game_over: bool) -> dict:
        """Print game summary and return results."""
        print("\n" + "=" * 60)
        print("GAME SUMMARY")
        print("=" * 60)
        print(f"Final Score: {self.score}/{self.max_score} ({100*self.score/self.max_score:.1f}%)")
        print(f"Steps Taken: {step}")
        print(f"Time Elapsed: {elapsed_time:.1f} seconds")
        print(f"Game Over: {game_over}")
        print("=" * 60)
        
        return {
            "final_score": self.score,
            "max_score": self.max_score,
            "score_percentage": 100 * self.score / self.max_score,
            "steps": step,
            "elapsed_time": elapsed_time,
            "game_over": game_over,
        }


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run the MCP ReAct agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the MCP ReAct Text Adventure Agent")
    parser.add_argument(
        "--server", "-s",
        default="mcp_server/zork_server.py",
        help="Path to the MCP server script"
    )
    parser.add_argument(
        "--max-steps", "-n",
        type=int,
        default=100,
        help="Maximum steps to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model to use"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    config = MCPAgentConfig(verbose=args.verbose)
    if args.model:
        config.model = args.model
    
    agent = MCPReActAgent(args.server, config)
    return await agent.run(max_steps=args.max_steps)


if __name__ == "__main__":
    asyncio.run(main())
