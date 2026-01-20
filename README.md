---
title: Agentic Zork
emoji: "🎮"
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Text Adventure LLM Agent Project

Build AI agents to play classic text adventure games (Zork, Colossal Cave, Enchanter, etc.) using the Model Context Protocol (MCP) and HuggingFace models.

## Overview

This project provides:

1. **MCP Server** - Exposes text adventure games as MCP tools using FastMCP
2. **ReAct Agent** - An agent that uses MCP tools to play games with reasoning
3. **Templates** - Starter code for students to implement their own solutions
4. **57 Games** - Zork trilogy, Infocom classics, and many more Z-machine games

## Architecture

```
+-------------------+     MCP Protocol     +------------------+
|                   | <------------------> |                  |
|   ReAct Agent     |    (tool calls)      |   MCP Server     |
|   (FastMCP Client)|                      |   (FastMCP)      |
|                   |                      |                  |
+-------------------+                      +------------------+
        |                                           |
        | LLM API                                   | Game API
        v                                           v
+-------------------+                      +------------------+
|   HuggingFace     |                      |   Text Adventure |
|   Inference API   |                      |   (Jericho)      |
+-------------------+                      +------------------+
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment (using uv recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your HuggingFace token (HF_TOKEN)
```

Get your HuggingFace token at: https://huggingface.co/settings/tokens

### 2. Run an Agent

```bash
# MCP mode (recommended) - uses FastMCP Client
python run_agent.py --mode mcp

# Basic ReAct agent (direct game interaction)
python run_agent.py --mode react

# Function calling mode
python run_agent.py --mode function --simple
```

## Project Structure

```
.
+-- run_agent.py              # Unified agent runner
+-- mcp_server/
|   +-- zork_server.py        # Full MCP server with all tools
+-- agents/
|   +-- base_agent.py         # Abstract base class
|   +-- react_agent.py        # Basic ReAct agent (no MCP)
|   +-- mcp_react_agent.py    # MCP-enabled ReAct agent
+-- templates/                # Student templates
|   +-- README.md             # Assignment instructions
|   +-- mcp_server_template.py    # MCP server starter
|   +-- react_agent_template.py   # Agent starter
+-- function_calling/         # Alternative: function calling
|   +-- controller.py
|   +-- simple_controller.py
|   +-- tools.py
+-- games/
|   +-- zork_env.py           # Jericho wrapper
+-- z-machine-games-master/   # Game files
```

## Agent Modes

| Mode | Description | Command |
|------|-------------|---------|
| `mcp` | MCP ReAct agent (FastMCP Client) | `--mode mcp` |
| `react` | Basic ReAct (direct game) | `--mode react` |
| `function` | Function calling (API) | `--mode function` |
| `function --simple` | Function calling (text) | `--mode function --simple` |

### Examples

```bash
# Run MCP agent with verbose output
python run_agent.py --mode mcp -v

# Run with different model
python run_agent.py --mode mcp --model google/gemma-2-2b-it

# Limit steps
python run_agent.py --mode mcp -n 50

# Play different games
python run_agent.py --mode mcp --game zork2
python run_agent.py --mode mcp --game advent     # Colossal Cave Adventure
python run_agent.py --mode mcp --game enchanter  # Infocom classic
python run_agent.py --mode mcp --game hhgg       # Hitchhiker's Guide

# List all 57 available games
python run_agent.py --list-games
```

## MCP Server Tools

The MCP server exposes these tools:

| Tool | Description |
|------|-------------|
| `play_action(action)` | Execute a game command (north, take lamp, etc.) |
| `memory()` | Get current state (location, score, history) |
| `get_map()` | View explored locations and connections |
| `inventory()` | Check items you're carrying |
| `valid_actions()` | Get command hints |
| `reset_game(game)` | Start over or switch games |
| `list_games()` | See all 57 available games |
| `hint()` | Get contextual hints |

### Testing the MCP Server

```bash
# Run server directly (stdio transport) - default game is zork1
python mcp_server/zork_server.py

# Run with a specific game
GAME=advent python mcp_server/zork_server.py

# Use MCP Inspector for interactive testing
npx @modelcontextprotocol/inspector python mcp_server/zork_server.py

# Use FastMCP dev mode
fastmcp dev mcp_server/zork_server.py
```

## Student Assignment

See [templates/README.md](templates/README.md) for the assignment.

Students implement:
1. **MCP Server** (`mcp_server_template.py`) - Expose game functionality as MCP tools
2. **ReAct Agent** (`react_agent_template.py`) - Play text adventures using MCP

## Configuration

### Environment Variables

Create `.env` from `.env.example`:

```bash
# Required: HuggingFace token
HF_TOKEN=hf_your_token_here

# Optional: Model override (default: meta-llama/Llama-3.2-3B-Instruct)
HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

### Recommended Models

| Model | Notes |
|-------|-------|
| `meta-llama/Llama-3.2-3B-Instruct` | Default, good balance |
| `google/gemma-2-2b-it` | Smaller, faster |
| `Qwen/Qwen2.5-7B-Instruct` | Good instruction following |

## Evaluation

Run the evaluator to test agent performance:

```bash
python evaluate.py --mode mcp --games zork1 --runs 3
```

Metrics:
- **Score**: Points earned in-game
- **Score %**: Score / Max possible score
- **Steps**: Number of actions taken
- **Time**: Elapsed time

## Resources

- [FastMCP Documentation](https://gofastmcp.com/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Jericho (Text Adventures)](https://github.com/microsoft/jericho)
- [HuggingFace Inference API](https://huggingface.co/docs/huggingface_hub/guides/inference)

## License

MIT
