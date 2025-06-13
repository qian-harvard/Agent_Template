import os
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
from pathlib import Path

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the simple chat agent with MCP support."""

    chat_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for the chat agent. Try 'openai:gpt-4' or 'anthropic:claude-3-sonnet' for better reasoning."
        },
    )

    temperature: float = Field(
        default=0.7,
        metadata={
            "description": "The temperature setting for the language model."
        },
    )

    # MCP Server Configuration
    mcp_servers: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "pandapower": {
                "command": "python",
                "args": [str(Path(__file__).parent.parent / "mcp" / "panda_mcp.py")],
                "transport": "stdio",
                "cwd": None,  # Will be set dynamically
                "env": None,
            }
        },
        metadata={
            "description": "Configuration for MCP servers to connect to."
        },
    )

    enable_mcp: bool = Field(
        default=True,
        metadata={
            "description": "Whether to enable MCP server integration."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
