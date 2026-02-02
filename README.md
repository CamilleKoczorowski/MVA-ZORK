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
hf_oauth: true
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

### 1. Download Game Files

The Z-machine game files are not included in this repository. Clone them from the BYU-PCCL repository:

```bash
git clone https://github.com/BYU-PCCL/z-machine-games z-machine-games-master
```

This provides 57 classic text adventure games including the Zork trilogy, Colossal Cave Adventure, Enchanter, and more.

### 2. Setup

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

### 3. Run an Agent

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
|   +-- evaluate.py           # Evaluation CLI (local and HF Spaces)
|   +-- runner.py             # Agent execution
|   +-- metrics.py            # Result tracking
+-- example_submission/       # Working example submission
|   +-- agent.py              # Full ReAct agent implementation
|   +-- mcp_server.py         # Full MCP server implementation
+-- submission_template/      # Student templates (HF Space template)
|   +-- README.md             # Assignment instructions
|   +-- agent.py              # Agent starter code
|   +-- mcp_server.py         # MCP server starter code
|   +-- app.py                # HF Spaces Gradio app
|   +-- requirements.txt      # Space dependencies
+-- games/
|   +-- zork_env.py           # Jericho wrapper
+-- z-machine-games-master/   # Game files
```

## Student Submission Workflow (Hugging Face Spaces)

### For Students

1. **Fork the template Space** on Hugging Face:
   ```
   https://huggingface.co/spaces/[COURSE]/text-adventure-template
   ```

2. **Clone your fork locally**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/text-adventure-agent
   cd text-adventure-agent
   ```

3. **Implement your agent** in `agent.py` and `mcp_server.py`

4. **Test locally**:
   ```bash
   # Test MCP server interactively
   fastmcp dev mcp_server.py
   
   # Run your agent
   python run_agent.py --agent . --game zork1 -v -n 20
   ```

5. **Push to your Space**:
   ```bash
   git add -A
   git commit -m "Implement my agent"
   git push
   ```

6. **Submit** your Space URL on the course platform

### For Instructors

Evaluate student submissions from their HF Spaces:

```bash
# Evaluate a single Space
python evaluation/evaluate.py --hf-space student1/text-adventure-agent -g zork1 -t 5

# Evaluate a local submission
python evaluation/evaluate.py -s ./example_submission -g zork1 -t 5

# Batch evaluate multiple local submissions
python evaluation/evaluate.py --submissions-dir ./all_submissions -g zork1 -o results.json
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