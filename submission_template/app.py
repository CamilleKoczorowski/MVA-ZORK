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


def read_readme():
    """Read the README content."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text()
    return "# Submission\n\nNo README.md found."


def read_file_content(filename: str) -> str:
    """Read a source file's content."""
    file_path = Path(__file__).parent / filename
    if file_path.exists():
        return file_path.read_text()
    return f"# File not found: {filename}"


# Create the Gradio interface
with gr.Blocks(title="Text Adventure Agent Submission") as demo:
    gr.Markdown("# Text Adventure Agent Submission")
    gr.Markdown(
        "This Space contains a student submission for the Text Adventure Agent assignment. "
        "Use the tabs below to view the submitted code."
    )
    
    with gr.Tabs():
        with gr.Tab("README"):
            gr.Markdown(read_readme())
        
        with gr.Tab("Agent Code"):
            gr.Code(
                value=read_file_content("agent.py"),
                language="python",
                label="agent.py",
            )
        
        with gr.Tab("MCP Server Code"):
            gr.Code(
                value=read_file_content("mcp_server.py"),
                language="python",
                label="mcp_server.py",
            )
    
    gr.Markdown(
        "---\n"
        "**Note:** This is a code submission Space. "
        "Evaluation is performed using the evaluation script."
    )


if __name__ == "__main__":
    demo.launch()
