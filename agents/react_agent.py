"""
ReAct Agent for Text Adventure Games

Implements a ReAct (Reasoning + Acting) loop using an LLM to play text adventures.
The agent thinks about its situation, decides on an action, and learns from the result.
"""

import os
from dataclasses import dataclass
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from agents.base_agent import BaseAgent, AgentConfig
from games.zork_env import GameState


@dataclass
class ReActConfig(AgentConfig):
    """Configuration for the ReAct agent."""
    name: str = "ReActAgent"
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 300
    max_history: int = 15


SYSTEM_PROMPT = """You are playing a classic text adventure game.

GOAL: Explore the world, solve puzzles, collect treasures, and maximize your score.

VALID COMMANDS:
- Movement: north, south, east, west, up, down, enter, exit
- Looking: look, examine <thing>, read <thing>  
- Objects: take <item>, drop <item>, open <thing>, close <thing>
- Light: turn on lamp, light match
- Combat: attack <enemy> with <weapon>
- Other: inventory, wait, push <thing>, move <thing>

INVALID COMMANDS (do NOT use): check, inspect, search, grab, use, help

TIPS:
- Explore systematically - try all directions
- Examine interesting objects and read documents
- Pick up useful items (lamp, keys, weapons)
- Open containers to find hidden items

You MUST respond in EXACTLY this format (no markdown, no extra text):
THOUGHT: <your reasoning in one sentence>
ACTION: <one valid command>

Example response:
THOUGHT: I see a container here, I should check what is inside.
ACTION: open container"""


class ReActAgent(BaseAgent):
    """
    A ReAct (Reasoning + Acting) agent that uses an LLM to play text adventures.
    
    Uses Hugging Face Hub's Inference API.
    """
    
    def __init__(self, config: ReActConfig = None, token: str = None):
        super().__init__(config or ReActConfig())
        self.config: ReActConfig = self.config
        
        # Load token from environment if not provided
        load_dotenv()
        token = token or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Set HF_TOKEN environment variable or pass token parameter.")
        
        # Override model from environment if set
        env_model = os.getenv("HF_MODEL")
        if env_model:
            self.config.model = env_model
        
        self.client = InferenceClient(token=token)
        self.thoughts: list[str] = []  # Store reasoning history
    
    def choose_action(self, observation: str, game_state: GameState) -> str:
        """
        Use the LLM to reason about the situation and choose an action.
        """
        # Build the prompt with context
        prompt = self._build_prompt(observation, game_state)
        
        # Call the LLM
        response = self._call_llm(prompt)
        
        # Parse the response
        thought, action = self._parse_response(response)
        
        # Store the thought for history
        self.thoughts.append(thought)
        
        if self.config.verbose:
            print(f"\n[Thought] {thought}")
            print(f"[Action] {action}")
        
        return action
    
    def _build_prompt(self, observation: str, game_state: GameState) -> str:
        """Build the prompt for the LLM with current context."""
        parts = []
        
        # Current status (compact for small models)
        parts.append(f"Score: {game_state.score}/{game_state.max_score} | Moves: {game_state.moves}")
        
        if game_state.inventory:
            parts.append(f"Inventory: {', '.join(game_state.inventory)}")
        
        # Recent history (only last 3 for small models)
        if self.history:
            parts.append("\nRecent:")
            recent_actions = []
            for action, obs, state in self.history[-3:]:
                obs_short = obs[:150] + "..." if len(obs) > 150 else obs
                parts.append(f"> {action}\n{obs_short}")
                recent_actions.append(action)
            
            # Warn about repeated actions
            if len(recent_actions) >= 2 and len(set(recent_actions)) == 1:
                parts.append(f"\n[WARNING: You've done '{recent_actions[0]}' multiple times. Try something different!]")
        
        # Current observation
        parts.append(f"\nNow:\n{observation}")
        parts.append("\nWhat do you do next? (Try a NEW action)")
        
        return "\n".join(parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the Hugging Face Inference API."""
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "THOUGHT: Error occurred, trying a safe action.\nACTION: look"
    
    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse the LLM response to extract thought and action."""
        thought = ""
        action = "look"  # Default fallback action
        
        lines = response.strip().split("\n")
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            
            if line_upper.startswith("THOUGHT:"):
                # Extract thought (may span multiple lines until ACTION)
                thought_parts = [line.split(":", 1)[1].strip()]
                for j in range(i + 1, len(lines)):
                    if lines[j].upper().strip().startswith("ACTION:"):
                        break
                    thought_parts.append(lines[j].strip())
                thought = " ".join(thought_parts).strip()
            
            elif line_upper.startswith("ACTION:"):
                action = line.split(":", 1)[1].strip().lower()
                # Clean up the action - remove quotes, markdown, and extra whitespace
                action = action.strip('"\'')
                # Remove markdown bold/italic markers
                action = action.replace("**", "").replace("*", "").replace("__", "").replace("_", " ")
                # Remove backticks
                action = action.replace("`", "")
                # Clean up whitespace
                action = " ".join(action.split())
                break
        
        # Validate action isn't empty
        if not action or action.isspace():
            action = "look"
        
        return thought, action
    
    def reset(self):
        """Reset the agent for a new game."""
        super().reset()
        self.thoughts = []
    
    def get_summary(self) -> str:
        """Get a summary of the agent's reasoning."""
        if not self.thoughts:
            return "No thoughts recorded yet."
        
        return "\n---\n".join(self.thoughts[-5:])


# Example usage and testing
if __name__ == "__main__":
    import sys
    from games.zork_env import TextAdventureEnv
    
    # Use command line arg or default to zork1
    game = sys.argv[1] if len(sys.argv) > 1 else "zork1"
    
    # Quick test
    config = ReActConfig(verbose=True)
    
    try:
        agent = ReActAgent(config)
        env = TextAdventureEnv(game)
        
        state = env.reset()
        print("=" * 50)
        print(f"{game.upper()} (using {agent.config.model})")
        print("=" * 50)
        print(state.observation)
        
        # Run a few steps
        for step in range(5):
            print(f"\n{'=' * 50}")
            print(f"Step {step + 1}")
            print("=" * 50)
            
            action = agent.choose_action(state.observation, state)
            print(f"\n> {action}")
            
            state = env.step(action)
            print(f"\n{state.observation}")
            print(f"\nScore: {state.score}/{state.max_score}")
            
            agent.update_history(action, state.observation, state)
            
            if state.done:
                print("\nGAME OVER!")
                break
    
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Make sure to set your HF_TOKEN in .env file")
