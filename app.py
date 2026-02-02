"""
Gradio App - Text Adventure AI Agent Assignment

A simple interface for the text adventure AI agent assignment.
"""

import gradio as gr
from huggingface_hub import HfApi
from datetime import datetime
import json

TITLE = "Playing Zork has never been so boring"

DESCRIPTION = """
Build AI agents to play classic text adventure games (Zork, Colossal Cave, Enchanter, etc.) using the Model Context Protocol (MCP) and HuggingFace models.

This project provides:
- **MCP Server** - Exposes text adventure games as MCP tools using FastMCP
- **ReAct Agent** - An agent that uses MCP tools to play games with reasoning
- **Submission Template** - Starter code for students to implement their own solutions
- **Evaluation System** - Deterministic evaluation with seeded runs
- **57 Games** - Zork trilogy, Infocom classics, and many more Z-machine games
"""

CLONE_INSTRUCTIONS = """
## Getting Started

### 0. Clone this space

```bash
git clone https://huggingface.co/spaces/LLM-course/Agentic-zork
```

This includes:
- run_agent.py: Script to run agents on text adventure games
- evaluation/: Evaluation scripts and utilities
- games/: Text adventure game environments
- submission_template/: Template for your agent submission
- example_submission/: Example agent implementation

### 1. Fork the template space

Fork the template space on Hugging Face:
```
https://huggingface.co/spaces/LLM-course/text-adventure-template
```

### 2. Clone your fork locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/text-adventure-agent
```

### 3. Implement your agent

Edit these files:
- `agent.py` - Your ReAct agent implementation (implement the `StudentAgent` class)
- `mcp_server.py` - Your MCP server implementation (add tools like `play_action`, `memory`, etc.)

### 4. Test locally

```bash
# Test MCP server interactively
fastmcp dev mcp_server.py

# Run your agent
python run_agent.py --agent . --game lostpig -v -n 20
```

### 5. Push to your space

### 6. Submit your space URL
"""

DATASET_REPO = "LLM-course/zork-submission"


def submit_space(space_url: str, profile: gr.OAuthProfile | None) -> str:
    """Submit a space URL to the dataset."""
    if profile is None:
        return "Please log in with your HuggingFace account first (button above)."
    
    if not space_url or not space_url.strip():
        return "Please enter your Space URL."
    
    space_url = space_url.strip()
    
    # Validate URL format
    if not ("huggingface.co/spaces/" in space_url or "hf.co/spaces/" in space_url):
        return "Invalid Space URL. It should look like: https://huggingface.co/spaces/username/space-name"
    
    username = profile.username
    
    try:
        api = HfApi()
        
        # Try to load existing submissions
        try:
            submissions_path = api.hf_hub_download(
                repo_id=DATASET_REPO,
                filename="submissions.json",
                repo_type="dataset",
            )
            with open(submissions_path, "r") as f:
                submissions = json.load(f)
        except Exception:
            submissions = {}
        
        # Update with new submission (overwrites previous for same user)
        submissions[username] = {
            "space_url": space_url,
            "submitted_at": datetime.now().isoformat(),
        }
        
        # Save back to dataset
        submissions_json = json.dumps(submissions, indent=2)
        api.upload_file(
            path_or_fileobj=submissions_json.encode(),
            path_in_repo="submissions.json",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Update submission for {username}",
        )
        
        return f"Submission successful! Space URL '{space_url}' recorded for user '{username}'."
        
    except Exception as e:
        return f"Error submitting: {str(e)}"


demo = gr.Blocks(title=TITLE)

with demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    gr.Markdown("---")
    gr.Markdown(CLONE_INSTRUCTIONS)
    
    # Submission section
    gr.LoginButton()
    space_input = gr.Textbox(
        label="Your Space URL",
        placeholder="https://huggingface.co/spaces/your-username/text-adventure-agent",
    )
    submit_btn = gr.Button("Click HERE to Submit", variant="primary")
    result_text = gr.Textbox(label="Status", interactive=False)
    
    submit_btn.click(
        fn=submit_space,
        inputs=[space_input],
        outputs=[result_text],
    )

if __name__ == "__main__":
    demo.launch()
