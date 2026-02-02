---
title: Agentic Zork
emoji: 🎮
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.3.0
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
3. **Submission Template** - Starter code for students to implement their own solutions
4. **Evaluation System** - Deterministic evaluation with seeded runs
5. **57 Games** - Zork trilogy, Infocom classics, and many more Z-machine games

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
# Run the example MCP agent
python run_agent.py

# Play a different game
python run_agent.py --game advent

# Verbose output
python run_agent.py -v

# Limit steps
python run_agent.py -n 50

# List all 57 games
python run_agent.py --list-games
```

## Project Structure

```
.
+-- run_agent.py              # Agent runner
+-- app.py                    # Gradio interface
+-- evaluation/               # Evaluation system
|   +-- evaluate.py           # Main CLI script
|   +-- runner.py             # Agent execution
|   +-- metrics.py            # Result tracking
+-- example_submission/       # Working example submission
|   +-- agent.py              # Full ReAct agent implementation
|   +-- mcp_server.py         # Full MCP server implementation
+-- submission_template/      # Student templates
|   +-- README.md             # Assignment instructions
|   +-- agent.py              # Agent starter code
|   +-- mcp_server.py         # MCP server starter code
|   +-- app.py                # HF Spaces app
+-- games/
|   +-- zork_env.py           # Jericho wrapper
+-- z-machine-games-master/   # Game files
```

## Assignment

See [submission_template/README.md](submission_template/README.md) for the assignment instructions.

You need to implement:
1. **MCP Server** (`mcp_server.py`) - Expose game functionality as MCP tools
2. **ReAct Agent** (`agent.py`) - Play text adventures using MCP tools

A working example is provided in `example_submission/`.

## Evaluation

Run the evaluator to test submissions:

```bash
# Evaluate a submission
python evaluation/evaluate.py -s ./submission_template -g zork1 -t 5

# Evaluate the example
python evaluation/evaluate.py -s ./example_submission -g zork1 -t 3

# Evaluate multiple games
python evaluation/evaluate.py -s ./example_submission -g zork1 advent enchanter -t 3

# Save results to JSON
python evaluation/evaluate.py -s ./example_submission -g zork1 -t 3 -o results.json
```

Metrics:
- **Score**: Points earned in-game (averaged over trials)
- **Score %**: Score / Max possible score
- **Steps**: Number of actions taken
- **Time**: Elapsed time

## Configuration

### Environment Variables

Create `.env` from `.env.example`:

```bash
# Required: HuggingFace token
HF_TOKEN=hf_your_token_here
```

### Fixed Model

All submissions use the same model for fairness:
- **Model**: `Qwen/Qwen2.5-72B-Instruct`
- **Temperature**: `0.0` (deterministic)
- **Seed**: Provided for reproducibility

## License

MIT