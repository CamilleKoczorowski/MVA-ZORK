"""
Student Agent for Text Adventure Games

This is your submission file. Implement the StudentAgent class to play
text adventure games using the MCP server you also implement.

Your agent should:
1. Connect to the MCP server via the provided client
2. Use the ReAct pattern (Thought -> Action -> Observation)
3. Call MCP tools to interact with the game
4. Maximize the game score within the step limit

Required method:
    async def run(self, client, game, max_steps, seed, verbose) -> RunResult

The 'client' is a FastMCP Client already connected to your MCP server.
Use it to call tools like: await client.call_tool("play_action", {"action": "look"})

Tips:
- Start by looking around and understanding your environment
- Keep track of visited locations to avoid loops
- Pick up useful items (lamp, sword, etc.)
- The seed parameter should be used to set your LLM's seed for reproducibility
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# =============================================================================
# LLM Configuration - DO NOT MODIFY
# =============================================================================

# Model to use (fixed for fair evaluation)
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Initialize the LLM client (uses HF_TOKEN from environment)
_hf_token = os.getenv("HF_TOKEN")
if not _hf_token:
    raise ValueError("HF_TOKEN not found. Set it in your .env file.")

LLM_CLIENT = InferenceClient(token=_hf_token)


def call_llm(prompt: str, system_prompt: str, seed: int, max_tokens: int = 300) -> str:
    """
    Call the LLM with the given prompt. Use this function in your agent.
    
    Args:
        prompt: The user prompt (current game state, history, etc.)
        system_prompt: The system prompt (instructions for the agent)
        seed: Random seed for reproducibility
        max_tokens: Maximum tokens in response (default: 300)
        
    Returns:
        The LLM's response text
        
    Example:
        response = call_llm(
            prompt="You are in a forest. What do you do?",
            system_prompt=SYSTEM_PROMPT,
            seed=42,
        )
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,  # Deterministic for reproducibility
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
# System Prompt - Customize this for your agent
# =============================================================================

SYSTEM_PROMPT = """You are playing a classic text adventure game.

GOAL: Explore the world, solve puzzles, and maximize your score.

AVAILABLE TOOLS (use via MCP):
- play_action: Execute a game command (north, take lamp, open mailbox, etc.)
- memory: Get current game state and history (if implemented)
- inventory: Check what you're carrying (if implemented)

VALID GAME COMMANDS for play_action:
- Movement: north, south, east, west, up, down, enter, exit
- Objects: take <item>, drop <item>, open <thing>, close <thing>, examine <thing>
- Other: look, inventory, read <thing>, turn on lamp

RESPOND IN THIS EXACT FORMAT (no markdown):
THOUGHT: <your reasoning about what to do next>
TOOL: <tool_name>
ARGS: <JSON arguments, e.g., {"action": "look"}>

Example:
THOUGHT: I should look around to see where I am.
TOOL: play_action
ARGS: {"action": "look"}
"""


# =============================================================================
# Student Agent - IMPLEMENT THIS CLASS
# =============================================================================

class StudentAgent:
    """
    Your ReAct agent implementation.
    
    TODO:
    1. Implement the run() method with the ReAct loop
    2. Parse LLM responses to extract tool calls
    3. Track state and avoid loops
    
    Use the provided call_llm() function to interact with the LLM.
    """
    
    def __init__(self):
        """Initialize your agent here."""
        # TODO: Initialize any state tracking you need
        # self.history = []
        # self.visited_locations = set()
        pass
    
    async def run(
        self,
        client,  # FastMCP Client connected to your MCP server
        game: str,
        max_steps: int,
        seed: int,
        verbose: bool = False,
    ) -> RunResult:
        """
        Run the agent for a game session.
        
        Args:
            client: FastMCP Client connected to your MCP server
            game: Name of the game being played (e.g., "zork1")
            max_steps: Maximum number of steps to take
            seed: Random seed for reproducibility (use for LLM calls)
            verbose: Whether to print detailed output
            
        Returns:
            RunResult with final score and statistics
        """
        # TODO: Implement your ReAct loop here
        #
        # Basic structure:
        # 1. Get initial observation (call play_action with "look")
        # 2. Loop for max_steps:
        #    a. Build prompt with current observation and history
        #    b. Call LLM to get thought and action
        #    c. Parse the response to extract tool and args
        #    d. Call the tool via client.call_tool(tool_name, args)
        #    e. Update history and state
        #    f. Check for game over
        # 3. Return RunResult with final statistics
        
        # Example of calling a tool:
        # result = await client.call_tool("play_action", {"action": "look"})
        # observation = result[0].text if result else "No response"
        
        # Example of calling the LLM:
        # response = call_llm(
        #     prompt="Current observation: " + observation,
        #     system_prompt=SYSTEM_PROMPT,
        #     seed=seed,
        # )
        
        # Placeholder implementation - replace with your code
        locations_visited = set()
        history = []
        final_score = 0
        moves = 0
        
        # TODO: Your implementation here
        # ...
        
        return RunResult(
            final_score=final_score,
            max_score=350,  # Zork1 max score, adjust if needed
            moves=moves,
            locations_visited=locations_visited,
            game_completed=False,
            history=history,
        )
    
    def _build_prompt(self, observation: str, history: list) -> str:
        """
        Build the prompt for the LLM.
        
        TODO: Implement this to create effective prompts
        """
        # TODO: Combine system prompt, history, and current observation
        pass
    
    def _parse_response(self, response: str) -> tuple[str, str, dict]:
        """
        Parse LLM response to extract thought, tool name, and arguments.
        
        TODO: Implement robust parsing
        
        Returns:
            Tuple of (thought, tool_name, args_dict)
        """
        # TODO: Parse the response format:
        # THOUGHT: ...
        # TOOL: ...
        # ARGS: {...}
        pass
    
    def _call_llm(self, prompt: str, system_prompt: str, seed: int) -> str:
        """
        Call the LLM with the given prompt.
        
        This is a convenience wrapper - you can also use call_llm() directly.
        """
        return call_llm(prompt, system_prompt, seed)


# =============================================================================
# For local testing
# =============================================================================

async def test_agent():
    """Test the agent locally."""
    from fastmcp import Client
    
    # Path to your MCP server
    server_path = "mcp_server.py"
    
    agent = StudentAgent()
    
    async with Client(server_path) as client:
        result = await agent.run(
            client=client,
            game="zork1",
            max_steps=10,
            seed=42,
            verbose=True,
        )
        
        print(f"\nFinal Score: {result.final_score}")
        print(f"Moves: {result.moves}")
        print(f"Locations: {result.locations_visited}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
