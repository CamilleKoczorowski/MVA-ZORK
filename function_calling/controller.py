"""
Function-Calling Controller for Zork (API-Based)

This controller uses the HuggingFace API's native function calling feature.
The model is given tool schemas and can call them via the tools API.

Model: Llama 3.2 3B Instruct (supports native function calling)

Compare with simple_controller.py which uses text-based "parsing" approach.
"""

import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from tools import ALL_TOOLS, set_game_state, add_to_history

# Add parent directory to path to import games module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.zork_env import ZorkEnvironment


# System prompt for the agent
SYSTEM_PROMPT = """You are playing Zork, a classic text adventure game.

## YOUR GOAL
Explore, collect treasures (bring them to the trophy case), and maximize your score.

## VALID COMMANDS (use ONLY these exact verbs)

Movement:
  north, south, east, west, up, down (or n, s, e, w, u, d)
  enter, exit, climb, cross, go <direction>

Looking:
  look, examine <thing>, look at <thing>, look in <thing>, read <thing>

Objects:
  take <item>, drop <item>, pick up <item>
  open <thing>, close <thing>, unlock <thing> with <key>
  put <item> in <container>, give <item> to <person>

Light:
  turn on lamp, turn off lamp, light match

Combat:
  attack <enemy> with <weapon>, kill <enemy> with <weapon>

Other:
  inventory (or i), wait (or z), score, save, restore
  push <thing>, pull <thing>, move <thing>, tie <rope> to <thing>
  eat <food>, drink <liquid>, wave <item>

## FORBIDDEN (these will NOT work):
  check, inspect, search, investigate, grab, pick, use, interact,
  go to, walk to, head to, travel, proceed

## YOUR TOOLS
  memory()    - See current state and recent actions
  get_map()   - See explored locations
  inventory() - Check what you're carrying

## RESPONSE FORMAT
When you want to take a game action, respond with:
  ACTION: <command>

Examples:
  ACTION: open mailbox
  ACTION: north
  ACTION: take lamp
  ACTION: examine leaflet"""


# Valid Zork command verbs for validation
VALID_VERBS = {
    "north", "south", "east", "west", "up", "down", "n", "s", "e", "w", "u", "d",
    "look", "l", "examine", "x", "read",
    "take", "get", "drop", "put", "give",
    "open", "close", "unlock", "lock",
    "turn", "light", "extinguish", "blow",
    "attack", "kill", "fight", "hit",
    "enter", "exit", "go", "climb", "jump",
    "inventory", "i", "wait", "z", "score",
    "move", "push", "pull", "tie", "untie",
    "eat", "drink", "smell", "touch", "rub",
    "wave", "raise", "lower", "pour",
    "say", "answer", "yes", "no",
    "pray", "odysseus", "echo", "hello",
}


def validate_action(action: str) -> str:
    """Validate and potentially fix an action."""
    action = action.strip().lower()
    if not action:
        return "look"
    
    verb = action.split()[0]
    
    if verb in VALID_VERBS:
        return action
    
    # Common corrections
    corrections = {
        "check": "examine",
        "inspect": "examine", 
        "search": "examine",
        "grab": "take",
        "pick": "take",
        "see": "look",
        "view": "look",
        "walk": "go",
    }
    
    if verb in corrections:
        return corrections[verb] + action[len(verb):]
    
    return "look"  # Default fallback


def build_tool_schemas():
    """Convert LangChain tools to OpenAI function schemas."""
    schemas = []
    for tool in ALL_TOOLS:
        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        schemas.append(schema)
    return schemas


def run_tool(tool_name: str) -> str:
    """Execute a tool by name and return its result."""
    for tool in ALL_TOOLS:
        if tool.name == tool_name:
            return tool.invoke({})
    return f"Unknown tool: {tool_name}"


class FunctionCallingController:
    """Controller using LLM API-based function calling."""
    
    def __init__(self, model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not set in environment")
        
        self.client = InferenceClient(token=token)
        self.model = os.getenv("HF_MODEL", model)
        self.tool_schemas = build_tool_schemas()
    
    def get_action(self, observation: str, game_state) -> str:
        """Get the next action from the LLM."""
        
        # Update tool state
        set_game_state(
            observation=observation,
            inventory=list(game_state.inventory) if game_state.inventory else [],
            score=game_state.score,
            moves=game_state.moves
        )
        
        # Build messages fresh each time (simpler than managing tool history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Game output:\n{observation}\n\nWhat do you do?"}
        ]
        
        # Allow up to 3 tool calls before requiring action
        for _ in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
                max_tokens=300,
            )
            
            message = response.choices[0].message
            
            # Check if model wants to use a tool
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                
                print(f"  [Tool] {tool_name}")
                tool_result = run_tool(tool_name)
                print(f"  {tool_result[:100]}...")
                
                # Add tool interaction to messages for next iteration
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"}
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
                
                # Continue to get the actual action
                continue
            
            # Model responded with text - extract action
            content = message.content or ""
            
            # Look for ACTION: in response
            if "ACTION:" in content.upper():
                for line in content.split('\n'):
                    if "ACTION:" in line.upper():
                        action = line.split(":", 1)[1].strip().lower()
                        validated = validate_action(action)
                        if validated:
                            return validated
                        else:
                            print(f"  [Warning] Invalid action '{action}', defaulting to 'look'")
                            return "look"
            
            # If no ACTION found, try to extract a command from the response
            content_lower = content.lower().strip()
            validated = validate_action(content_lower)
            if validated:
                return validated
            
            # Default
            return "look"
        
        # After 3 tool calls, just return look
        return "look"


def main():
    """Run the API-based function-calling controller."""
    print("=" * 60)
    print("Zork - API Function Calling Controller")
    print("   (using Llama 3.2 3B with native tool calling)")
    print("=" * 60)
    
    controller = FunctionCallingController()
    env = ZorkEnvironment("zork1")
    
    state = env.reset()
    print(f"\n{state.observation}\n")
    
    max_steps = 30
    
    for step in range(max_steps):
        print(f"\n{'─' * 50}")
        print(f"Step {step + 1}/{max_steps} | Score: {state.score}")
        print("─" * 50)
        
        action = controller.get_action(state.observation, state)
        print(f"\n> ACTION: {action}")
        
        # Take action in game
        state = env.step(action)
        add_to_history(action, state.observation)
        
        print(f"\n{state.observation}")
        
        if state.reward > 0:
            print(f"\n+{state.reward} points!")
        
        if state.done:
            print("\nGAME OVER!")
            break
    
    print(f"\n{'=' * 60}")
    print(f"Final Score: {state.score}")
    print("=" * 60)


if __name__ == "__main__":
    main()
