# Zork MCP Server

This directory contains an MCP (Model Context Protocol) server that exposes Zork game tools to LLM agents.

## Overview

The MCP server wraps the Jericho Zork environment and provides tools that any MCP-compatible agent (like Mini SWE Agent) can use to play the game.

## Tools Available

| Tool | Description |
|------|-------------|
| `play_action(action)` | Execute a game command (e.g., "north", "take lamp") |
| `memory()` | Get current state summary (location, score, recent actions) |
| `get_map()` | View explored locations and connections |
| `inventory()` | Check items you're carrying |
| `valid_actions()` | Get hints on available commands |
| `reset_game(game)` | Start over with zork1, zork2, or zork3 |
| `hint()` | Get contextual hints for your situation |

## Resources

The server also exposes MCP resources:
- `zork://state` - Current game state
- `zork://history` - Complete action history  
- `zork://map` - Explored locations map

## Running the Server

### Standalone (for testing)
```bash
python mcp_server/zork_server.py
```

### With MCP Inspector (for debugging)
```bash
npx @modelcontextprotocol/inspector python mcp_server/zork_server.py
```

### With Mini SWE Agent
```bash
python play_zork.py
```

## Configuration

The `mcp_config.json` file configures the server for use with MCP clients:

```json
{
  "mcpServers": {
    "zork": {
      "command": "python",
      "args": ["mcp_server/zork_server.py"]
    }
  }
}
```

## Architecture

```
┌─────────────────────────────────────────┐
│         MCP Client (Agent)              │
│   (Mini SWE Agent / Claude / etc.)      │
└──────────────────┬──────────────────────┘
                   │ MCP Protocol (stdio)
                   ▼
┌─────────────────────────────────────────┐
│         Zork MCP Server                 │
│   (FastMCP - zork_server.py)            │
│                                         │
│   Tools: play_action, memory, map,      │
│          inventory, valid_actions,      │
│          reset_game, hint               │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│     Jericho + Frotz                     │
│   (Z-machine game interpreter)          │
└─────────────────────────────────────────┘
```