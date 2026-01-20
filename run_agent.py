#!/usr/bin/env python3
"""
Unified Text Adventure Agent Runner

Run different types of LLM agents to play text adventure games:
  - react:     Basic ReAct agent with HuggingFace models
  - function:  Function-calling controller (API-based or text-based)
  - mcp:       MCP ReAct agent using FastMCP Client

Usage:
    python run_agent.py --mode react
    python run_agent.py --mode function
    python run_agent.py --mode mcp

Examples:
    # Run the basic ReAct agent
    python run_agent.py --mode react

    # Run the function-calling controller (API-based)
    python run_agent.py --mode function

    # Run the function-calling controller (text-based, works with any model)
    python run_agent.py --mode function --simple

    # Run with MCP ReAct agent (uses FastMCP Client)
    python run_agent.py --mode mcp
    
    # Play a different game
    python run_agent.py --mode mcp --game advent
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add games module to path for discovering available games
sys.path.insert(0, str(Path(__file__).parent))
from games.zork_env import list_available_games, TextAdventureEnv


# =============================================================================
# Mode: ReAct Agent
# =============================================================================

def run_react_agent(args):
    """Run the basic ReAct agent."""
    from agents.react_agent import ReActAgent, ReActConfig
    
    print("\n[ReAct] Running ReAct Agent")
    print(f"   Game: {args.game}")
    print(f"   Model: {args.model}")
    print()
    
    env = TextAdventureEnv(args.game)
    config = ReActConfig(verbose=args.verbose, model=args.model)
    agent = ReActAgent(config)
    
    return run_game_loop(env, agent, args.max_steps, args.verbose)


def run_game_loop(env, agent, max_steps: int, verbose: bool) -> dict:
    """Common game loop for ReAct-style agents."""
    state = env.reset()
    agent.reset()
    
    print("=" * 60)
    print(f"{env.game.upper()} - Starting Game")
    print(f"Max Score: {state.max_score}")
    print("=" * 60)
    print(f"\n{state.observation}\n")
    
    start_time = time.time()
    step = 0
    
    try:
        for step in range(1, max_steps + 1):
            print(f"\n{'─' * 40}")
            print(f"Step {step}")
            print("─" * 40)
            
            action = agent.choose_action(state.observation, state)
            print(f"\n> {action}")
            
            state = env.step(action)
            print(f"\n{state.observation}")
            
            if state.reward > 0:
                print(f"\n+{state.reward} points! (Total: {state.score}/{state.max_score})")
            elif state.reward < 0:
                print(f"\n{state.reward} points! (Total: {state.score}/{state.max_score})")
            else:
                print(f"\nScore: {state.score}/{state.max_score}")
            
            agent.update_history(action, state.observation, state)
            
            if state.done:
                print("\n" + "=" * 60)
                print("GAME OVER!")
                break
    
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")
    
    elapsed_time = time.time() - start_time
    return print_summary(env.game, state, step, elapsed_time)


# =============================================================================
# Mode: MCP ReAct Agent
# =============================================================================

def run_mcp_agent(args):
    """Run MCP ReAct Agent using FastMCP Client."""
    import asyncio
    from agents.mcp_react_agent import MCPReActAgent, MCPAgentConfig
    
    print("\n[MCP] Running MCP ReAct Agent with FastMCP")
    print(f"   Game: {args.game}")
    print(f"   Model: {args.model}")
    print(f"   Server: mcp_server/zork_server.py")
    print()
    
    config = MCPAgentConfig(verbose=args.verbose, model=args.model, game=args.game)
    agent = MCPReActAgent("mcp_server/zork_server.py", config)
    
    return asyncio.run(agent.run(max_steps=args.max_steps))


# =============================================================================
# Mode: Function Calling
# =============================================================================

def run_function_calling(args):
    """Run the function-calling controller."""
    # Import the appropriate controller
    sys.path.insert(0, str(Path(__file__).parent / "function_calling"))
    from tools import add_to_history
    
    if args.simple:
        from simple_controller import SimpleController
        print("\n[Function] Running Function Calling Controller (text-based)")
        controller = SimpleController(model=args.model)
    else:
        from controller import FunctionCallingController
        print("\n[Function] Running Function Calling Controller (API-based)")
        controller = FunctionCallingController(model=args.model)
    
    print(f"   Game: {args.game}")
    print(f"   Model: {args.model}")
    print()
    
    env = TextAdventureEnv(args.game)
    state = env.reset()
    
    print("=" * 60)
    print(f"{args.game.upper()} - Function Calling Mode")
    print("=" * 60)
    print(f"\n{state.observation}\n")
    
    start_time = time.time()
    step = 0
    
    try:
        for step in range(1, args.max_steps + 1):
            print(f"\n{'─' * 50}")
            print(f"Step {step}/{args.max_steps} | Score: {state.score}")
            print("─" * 50)
            
            action = controller.get_action(state.observation, state)
            print(f"\n> ACTION: {action}")
            
            state = env.step(action)
            add_to_history(action, state.observation)
            
            print(f"\n{state.observation}")
            
            if state.reward > 0:
                print(f"\n+{state.reward} points!")
            
            if state.done:
                print("\nGAME OVER!")
                break
    
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")
    
    elapsed_time = time.time() - start_time
    return print_summary(args.game, state, step, elapsed_time)


# =============================================================================
# Common Utilities
# =============================================================================

def print_summary(game: str, state, step: int, elapsed_time: float) -> dict:
    """Print game summary and return results dict."""
    print("\n" + "=" * 60)
    print("GAME SUMMARY")
    print("=" * 60)
    print(f"Game: {game}")
    print(f"Final Score: {state.score}/{state.max_score} ({100*state.score/state.max_score:.1f}%)")
    print(f"Total Moves: {state.moves}")
    print(f"Steps Taken: {step}")
    print(f"Time Elapsed: {elapsed_time:.1f} seconds")
    print("=" * 60)
    
    return {
        "game": game,
        "final_score": state.score,
        "max_score": state.max_score,
        "score_percentage": 100 * state.score / state.max_score,
        "moves": state.moves,
        "steps": step,
        "elapsed_time": elapsed_time,
        "game_over": state.done,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run an LLM agent to play text adventure games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  react     Basic ReAct agent (direct game interaction)
  function  Function-calling controller (use --simple for text-based)
  mcp       MCP ReAct agent using FastMCP Client (recommended)

Examples:
  python run_agent.py --mode react
  python run_agent.py --mode function
  python run_agent.py --mode function --simple  # text-based, any model
  python run_agent.py --mode mcp                # MCP with FastMCP
  python run_agent.py --mode mcp --game advent  # Play different game
  python run_agent.py --mode mcp --model google/gemma-2-2b-it
        """
    )
    
    # Get available games for help text
    available_games = list_available_games()
    game_help = f"Game to play (default: zork1). {len(available_games)} games available."
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="react",
        choices=["react", "function", "mcp"],
        help="Which agent mode to use (default: react)"
    )
    parser.add_argument(
        "--game", "-g",
        type=str,
        default="zork1",
        help=game_help
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="List all available games and exit"
    )
    parser.add_argument(
        "--max-steps", "-n",
        type=int,
        default=100,
        help="Maximum number of steps to run (default: 100)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed reasoning from the agent"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use text-based function calling (works with any model, only for --mode function)"
    )
    
    args = parser.parse_args()
    
    # Handle --list-games
    if args.list_games:
        print(f"\nAvailable games ({len(available_games)} total):\n")
        # Print in columns
        cols = 5
        for i in range(0, len(available_games), cols):
            row = available_games[i:i+cols]
            print("  " + "  ".join(f"{g:<15}" for g in row))
        print()
        sys.exit(0)
    
    # Validate game choice
    if args.game.lower() not in available_games:
        print(f"\nError: Unknown game '{args.game}'")
        print(f"Use --list-games to see {len(available_games)} available options.")
        sys.exit(1)
    
    # Get default model from environment
    default_model = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    
    # Set model if not specified
    if args.model is None:
        args.model = default_model
    
    print("\n" + "=" * 60)
    print("Text Adventure LLM Agent Runner")
    print("=" * 60)
    print(f"Mode: {args.mode}" + (" (simple)" if args.simple else ""))
    print(f"Game: {args.game}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Model: {args.model}")
    print(f"Verbose: {args.verbose}")
    
    # Run the selected mode
    try:
        if args.mode == "react":
            results = run_react_agent(args)
        elif args.mode == "function":
            results = run_function_calling(args)
        elif args.mode == "mcp":
            results = run_mcp_agent(args)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[Error] {e}")
        print("\nTo fix this:")
        print("1. Copy .env.example to .env")
        print("2. Add your HuggingFace token (HF_TOKEN)")
        sys.exit(1)
    except ImportError as e:
        print(f"\n[Import Error] {e}")
        print("\nMake sure to install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    return results


if __name__ == "__main__":
    main()
