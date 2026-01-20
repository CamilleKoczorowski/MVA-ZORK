"""
Function-Calling Controller for Zork (Text-Based)

This controller uses text-based "function calling" - the LLM outputs
TOOL: <name> or ACTION: <command> and we parse the text response.

Model: Qwen 2.5 7B Instruct (any chat model works)

This approach is:
- Simpler and more reliable than API-based function calling
- Works with any chat model (no special support needed)

Compare with controller.py which uses API-based tool calling.
"""

import os
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from tools import ALL_TOOLS, set_game_state, add_to_history

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.zork_env import ZorkEnvironment


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
  TOOL: memory    - See current state and recent actions
  TOOL: get_map   - See explored locations
  TOOL: inventory - Check what you're carrying

## RESPONSE FORMAT
Either use a tool:
  TOOL: memory

Or take a game action:
  ACTION: open mailbox

Always respond with TOOL: or ACTION: followed by your choice."""


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


def run_tool(tool_name: str) -> str:
    """Execute a tool by name."""
    tool_name = tool_name.strip().lower().replace(" ", "_")
    for tool in ALL_TOOLS:
        if tool.name == tool_name:
            return tool.invoke({})
    return f"Unknown tool: {tool_name}. Available: memory, get_map, inventory"


class SimpleController:
    """Controller using text-based tool calling."""
    
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not set in environment")
        
        self.client = InferenceClient(token=token)
        self.model = os.getenv("HF_MODEL", model)
        self.messages = []
    
    def _call_llm(self, user_message: str) -> str:
        """Call the LLM and get response."""
        self.messages.append({"role": "user", "content": user_message})
        
        # Keep conversation short
        if len(self.messages) > 15:
            self.messages = self.messages[-15:]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.messages,
            max_tokens=150,
            temperature=0.7,
        )
        
        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply
    
    def _validate_action(self, action: str) -> str | None:
        """Validate and potentially fix an action. Returns None if invalid."""
        action = action.strip().lower()
        if not action:
            return None
        
        # Get the first word (verb)
        verb = action.split()[0]
        
        # Check if it's a valid verb
        if verb in VALID_VERBS:
            return action
        
        # Try common corrections
        corrections = {
            "check": "examine",
            "inspect": "examine",
            "search": "examine",
            "grab": "take",
            "pick": "take",  # "pick up" -> "take"
            "see": "look",
            "view": "look",
            "walk": "go",
        }
        
        if verb in corrections:
            fixed = corrections[verb] + action[len(verb):]
            print(f"  [Correcting] '{verb}' -> '{corrections[verb]}'")
            return fixed
        
        return None
    
    def get_action(self, observation: str, game_state) -> str:
        """Get the next action, allowing tool use."""
        
        # Update tool state
        set_game_state(
            observation=observation,
            inventory=list(game_state.inventory) if game_state.inventory else [],
            score=game_state.score,
            moves=game_state.moves
        )
        
        prompt = f"Game:\n{observation}\n\nRespond with TOOL: or ACTION:"
        
        # Allow up to 3 tool calls before requiring an action
        for _ in range(3):
            response = self._call_llm(prompt)
            
            # Check for TOOL:
            tool_match = re.search(r'TOOL:\s*(\w+)', response, re.IGNORECASE)
            if tool_match:
                tool_name = tool_match.group(1)
                print(f"  [Tool] {tool_name}")
                
                result = run_tool(tool_name)
                print(f"  {result[:80]}...")
                
                # Feed result back
                prompt = f"Tool result:\n{result}\n\nNow respond with TOOL: or ACTION:"
                continue
            
            # Check for ACTION:
            action_match = re.search(r'ACTION:\s*(.+)', response, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().lower()
                # Clean up action (remove quotes, extra text)
                action = action.split('\n')[0].strip('"\'')
                
                # Validate the action
                validated = self._validate_action(action)
                if validated:
                    return validated
                else:
                    print(f"  [Warning] Invalid action '{action}', asking for retry...")
                    prompt = f"'{action}' is not a valid Zork command. Use verbs like: look, examine, take, open, north, south, etc.\n\nRespond with ACTION:"
                    continue
            
            # If neither, try to extract a command
            words = response.lower().split()
            for cmd in ["north", "south", "east", "west", "up", "down", 
                       "look", "take", "open", "enter", "examine"]:
                if cmd in words:
                    idx = words.index(cmd)
                    return " ".join(words[idx:idx+3])
            
            return "look"
        
        return "look"


def main():
    """Run the simple controller."""
    print("=" * 60)
    print("Zork - Simple Function Calling Demo")
    print("=" * 60)
    
    controller = SimpleController()
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
