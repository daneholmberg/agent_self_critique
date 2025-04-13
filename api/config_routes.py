from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

from agents.manim_agent import config as manim_agent_config


router = APIRouter(tags=["Configuration"])


class ManimConfigDefaults(BaseModel):
    defaultGeneralContext: str
    defaultFinalCommand: str


@router.get("/defaults/manim", response_model=ManimConfigDefaults)
async def get_manim_config_defaults():
    """Returns the default general context and final command prompts for the Manim agent UI."""
    # Check if the import succeeded at startup
    if manim_agent_config is None:
        raise HTTPException(
            status_code=404, detail="Manim agent configuration module could not be imported."
        )

    try:
        # Access the already imported module
        defaults = ManimConfigDefaults(
            defaultGeneralContext=getattr(
                manim_agent_config, "DEFAULT_GENERATOR_GENERAL_CONTEXT", ""
            ).strip(),
            defaultFinalCommand=getattr(
                manim_agent_config, "DEFAULT_GENERATOR_FINAL_COMMAND", ""
            ).strip(),
        )
        return defaults
    except AttributeError as e:
        # getattr already handles missing attributes gracefully by returning '',
        # but we keep this catch just in case other unexpected AttributeErrors occur.
        # More likely, a different error type would occur if the module structure is wrong.
        raise HTTPException(
            status_code=500, detail=f"Error accessing config value in Manim agent: {e}"
        )
    except Exception as e:
        # Catch any other unexpected errors
        # Log this error server-side for debugging
        print(f"ERROR loading Manim defaults: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load Manim agent config defaults: {e}"
        )


# You can add more endpoints here for other agents or general config if needed
# Example:
# @router.get("/defaults/another_agent")
# async def get_another_agent_defaults():
#     # ... load config for another_agent ...
#     pass
