from .zork_env import TextAdventureEnv, GameState, list_available_games, discover_games

# Alias for backwards compatibility
ZorkEnvironment = TextAdventureEnv

__all__ = ["TextAdventureEnv", "ZorkEnvironment", "GameState", "list_available_games", "discover_games"]
