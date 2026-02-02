# Example: MCP ReAct Agent

This is a complete, working example submission that demonstrates a ReAct agent using MCP.

## Approach

This agent uses the full ReAct pattern:
1. **Thought**: Reason about the current situation
2. **Tool**: Choose and call an MCP tool
3. **Observation**: Process the result

Features:
- Loop detection (avoids repeating the same action)
- Action validation (fixes common invalid verbs)
- Score tracking
- History management

## Files

- `agent.py` - ReAct agent with full implementation
- `mcp_server.py` - MCP server with memory, map, and inventory tools

## Testing

```bash
# Test locally
python agent.py
```
