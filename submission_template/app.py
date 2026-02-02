"""
Hugging Face Space - Text Adventure Agent Submission

This is a code-only Space for submitting your agent implementation.
The evaluation is run separately.

Files in this submission:
- agent.py: Your ReAct agent implementation
- mcp_server.py: Your MCP server implementation
- requirements.txt: Additional dependencies

To test locally:
    fastmcp dev mcp_server.py
    python agent.py
"""

import gradio as gr
from pathlib import Path

# Create the Gradio interface
with gr.Blocks(title="Text Adventure Agent Submission") as demo:
    gr.Markdown("# Text Adventure Agent Submission")
    gr.Markdown(
        "This Space contains a template submission for the Text Adventure Agent assignment. "
    )
        
    gr.Markdown(
        "---\n"
        "**Note:** This is a code submission Space. "
        "Evaluation is performed using the evaluation script.\n\n"
        "[Back to main assignment page](https://huggingface.co/spaces/LLM-course/Agentic-zork)"
    )


if __name__ == "__main__":
    demo.launch()
