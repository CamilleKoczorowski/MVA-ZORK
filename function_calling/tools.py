"""
Simple tools for the Zork agent using LangChain's tool decorator.
"""

from langchain_core.tools import tool


# Game state that tools can access (set by the controller)
_game_state = {
    "observation": "",
    "inventory": [],
    "score": 0,
    "moves": 0,
    "history": [],  # List of (action, result) tuples
}


def set_game_state(observation: str, inventory: list, score: int, moves: int):
    """Update the game state (called by controller after each action)."""
    _game_state["observation"] = observation
    _game_state["inventory"] = inventory
    _game_state["score"] = score
    _game_state["moves"] = moves


def add_to_history(action: str, result: str):
    """Add an action and its result to history."""
    _game_state["history"].append((action, result))
    # Keep only last 10 actions
    if len(_game_state["history"]) > 10:
        _game_state["history"] = _game_state["history"][-10:]


@tool
def memory() -> str:
    """Get a summary of the current game state including location, score, and recent actions."""
    obs = _game_state["observation"]
    score = _game_state["score"]
    moves = _game_state["moves"]
    
    # Extract location (first line of observation)
    lines = obs.strip().split('\n')
    location = lines[0] if lines else "Unknown"
    
    # Recent actions
    recent = _game_state["history"][-5:] if _game_state["history"] else []
    recent_str = "\n".join([f"  > {a} → {r[:50]}..." for a, r in recent]) if recent else "  (none yet)"
    
    return f"""Current State:
- Location: {location}
- Score: {score} points
- Moves: {moves}

Recent Actions:
{recent_str}

Current Observation:
{obs}"""


@tool
def get_map() -> str:
    """Get a map showing known locations and connections based on exploration history."""
    # Build a simple map from history
    locations = set()
    connections = []
    
    prev_loc = None
    for action, result in _game_state["history"]:
        # Extract location from result
        lines = result.strip().split('\n')
        if lines:
            loc = lines[0]
            locations.add(loc)
            
            # If this was a movement action, record connection
            if action in ["north", "south", "east", "west", "up", "down", "enter", "exit"]:
                if prev_loc and prev_loc != loc:
                    connections.append(f"  {prev_loc} --{action}--> {loc}")
                prev_loc = loc
    
    if not locations:
        return "Map: No locations explored yet. Try moving around!"
    
    loc_list = "\n".join([f"  - {loc}" for loc in sorted(locations)])
    conn_list = "\n".join(connections[-10:]) if connections else "  (no connections recorded)"
    
    return f"""Known Locations:
{loc_list}

Connections:
{conn_list}"""


@tool
def inventory() -> str:
    """Get the list of items currently in your inventory."""
    items = _game_state["inventory"]
    
    if not items:
        return "Inventory: You are empty-handed."
    
    # Clean up item names (Jericho returns objects with metadata)
    item_names = []
    for item in items:
        item_str = str(item)
        # Handle Jericho's object format: "leaflet Parent4 Sibling0..."
        # Look for "Parent" (case-insensitive) to find where metadata starts
        item_lower = item_str.lower()
        if "parent" in item_lower:
            idx = item_lower.index("parent")
            name = item_str[:idx].strip()
            # Remove leading "obj123: " if present
            if ":" in name:
                name = name.split(":", 1)[1].strip()
            item_names.append(name)
        elif ":" in item_str:
            name = item_str.split(":")[1].strip()
            item_names.append(name)
        else:
            item_names.append(item_str)
    
    return f"Inventory: {', '.join(item_names)}"


# Export all tools
ALL_TOOLS = [memory, get_map, inventory]
