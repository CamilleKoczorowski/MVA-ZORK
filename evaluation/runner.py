"""
Agent Runner for Evaluation

Handles spawning the MCP server subprocess and running the agent.
Provides isolation between trials and proper cleanup.
"""

import asyncio
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from games.zork_env import list_available_games


@dataclass
class RunConfig:
    """Configuration for a single agent run."""
    agent_path: Path
    server_path: Path
    game: str
    max_steps: int
    seed: int
    verbose: bool = False


@dataclass 
class RunResult:
    """Result of a single agent run."""
    final_score: int
    max_score: int
    moves: int
    locations_visited: set[str]
    game_completed: bool
    error: Optional[str] = None
    history: list[tuple[str, str, str]] = None  # (thought, action, result)

    def __post_init__(self):
        if self.history is None:
            self.history = []


def load_agent_class(agent_path: Path):
    """
    Dynamically load the agent class from student's agent.py.
    
    Expects the student file to define a class called 'StudentAgent'
    with an async method 'run(client, game, max_steps, seed)'.
    """
    spec = importlib.util.spec_from_file_location("student_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    
    # Add the submission directory to path so relative imports work
    submission_dir = str(agent_path.parent)
    if submission_dir not in sys.path:
        sys.path.insert(0, submission_dir)
    
    spec.loader.exec_module(module)
    
    if not hasattr(module, "StudentAgent"):
        raise ValueError(
            f"Agent file {agent_path} must define a 'StudentAgent' class"
        )
    
    return module.StudentAgent


async def run_agent_with_server(config: RunConfig) -> RunResult:
    """
    Run the student's agent with their MCP server.
    
    1. Spawns the MCP server as a subprocess
    2. Connects the agent via FastMCP Client
    3. Runs the agent for max_steps
    4. Collects and returns results
    """
    # Validate paths
    if not config.agent_path.exists():
        return RunResult(
            final_score=0,
            max_score=0,
            moves=0,
            locations_visited=set(),
            game_completed=False,
            error=f"Agent file not found: {config.agent_path}"
        )
    
    if not config.server_path.exists():
        return RunResult(
            final_score=0,
            max_score=0,
            moves=0,
            locations_visited=set(),
            game_completed=False,
            error=f"Server file not found: {config.server_path}"
        )
    
    # Validate game
    available_games = list_available_games()
    if config.game not in available_games:
        return RunResult(
            final_score=0,
            max_score=0,
            moves=0,
            locations_visited=set(),
            game_completed=False,
            error=f"Unknown game: {config.game}. Available: {available_games[:10]}..."
        )
    
    try:
        # Load the student's agent class
        AgentClass = load_agent_class(config.agent_path)
        agent = AgentClass()
        
        # Create transport for the MCP server
        # Set environment variable for the game
        env = os.environ.copy()
        env["GAME"] = config.game
        
        transport = StdioTransport(
            command=sys.executable,
            args=[str(config.server_path)],
            env=env,
        )
        
        # Connect to the server and run the agent
        async with Client(transport) as client:
            result = await agent.run(
                client=client,
                game=config.game,
                max_steps=config.max_steps,
                seed=config.seed,
                verbose=config.verbose,
            )
            
            return result
            
    except Exception as e:
        import traceback
        return RunResult(
            final_score=0,
            max_score=0,
            moves=0,
            locations_visited=set(),
            game_completed=False,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )


async def run_reference_agent(
    game: str,
    max_steps: int,
    seed: int,
    verbose: bool = False,
) -> RunResult:
    """
    Run the reference agent (from example_submission) for baseline comparison.
    """
    # Use the example as the reference
    examples_dir = Path(__file__).parent.parent / "example_submission"
    agent_path = examples_dir / "agent.py"
    server_path = examples_dir / "mcp_server.py"
    
    config = RunConfig(
        agent_path=agent_path,
        server_path=server_path,
        game=game,
        max_steps=max_steps,
        seed=seed,
        verbose=verbose,
    )
    
    return await run_agent_with_server(config)


def run_single_trial(config: RunConfig) -> RunResult:
    """Synchronous wrapper for running a single trial."""
    return asyncio.run(run_agent_with_server(config))
