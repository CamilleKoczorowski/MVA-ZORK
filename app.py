"""
Gradio App - Text Adventure AI Agent Assignment

A simple interface for the text adventure AI agent assignment.
"""

import gradio as gr

TITLE = "Playing Zork has never been so boring"

DESCRIPTION = """
In this assignment, you will build an AI Agent and an MCP server to play text adventure games like Zork.

The evaluation server is not ready yet, but you can look at the templates by cloning this repository.
"""

CLONE_INSTRUCTIONS = """
## Getting Started

### 1. Clone the Repository

```bash
git clone https://huggingface.co/spaces/LLM-course/Agentic-zork
cd Agentic-zork
```

### 2. Set Up Environment

```bash
# Create virtual environment (using uv recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Configure API Token

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your HuggingFace token
# HF_TOKEN=hf_your_token_here
```

Get your HuggingFace token at: https://huggingface.co/settings/tokens

### 4. Explore the Templates

The submission template is in the `submission_template/` folder:

- `agent.py` - Your agent implementation (implement the StudentAgent class)
- `mcp_server.py` - Your MCP server implementation (add tools)
- `README.md` - Detailed instructions

A working example is in `examples/mcp_react/`.

### 5. Test Your Implementation

```bash
# Run the example agent
python run_agent.py

# Run with a different game
python run_agent.py --game advent

# List available games (57 total!)
python run_agent.py --list-games
```

## Resources

- [Submission Instructions](submission_template/README.md)
- [FastMCP Documentation](https://gofastmcp.com/)
- [MCP Protocol](https://modelcontextprotocol.io/)
"""

demo = gr.Blocks(title=TITLE)

with demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    gr.Markdown("---")
    gr.Markdown(CLONE_INSTRUCTIONS)

if __name__ == "__main__":
    demo.launch()
