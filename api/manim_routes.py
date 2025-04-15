import os
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime

# Assuming SceneIdeator is correctly located and importable
# Adjust the import path based on your project structure if needed
from agents.manim_agent.components.scene_ideator import SceneIdeator

router = APIRouter()


# Pydantic Models for Request and Response
class IdeationRequest(BaseModel):
    input_text: str
    other_info: Optional[str] = None
    initial_idea: Optional[str] = None
    num_ideas: Optional[int] = None
    # Consider adding num_ideas if you want the client to specify it
    # num_ideas: Optional[int] = None


class IdeationResponse(BaseModel):
    generated_ideas: List[str]


# Initialize the SceneIdeator component
# We can potentially configure num_ideas from config later if needed
# For now, using the default number of ideas from SceneIdeator's init
ideator = SceneIdeator()


@router.post("/ideate", response_model=IdeationResponse, tags=["Manim Agent"])
async def generate_ideas(request: IdeationRequest):
    """
    Receives input text, optional other info, and an optional initial idea,
    then uses the SceneIdeator to generate multiple scene concepts.
    """
    try:
        print(
            f"Received ideation request: input='{request.input_text[:50]}...', other='{str(request.other_info)[:50]}...', initial='{str(request.initial_idea)[:50]}...'"
        )

        # Create a run output directory for this ideation request
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = Path("outputs") / f"ideation_{timestamp}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Call the ideator's ideate method with run_output_dir
        result: Dict[str, Any] = await ideator.ideate(
            input_text=request.input_text,
            other_info=request.other_info,
            initial_idea=request.initial_idea,
            num_ideas=request.num_ideas,
            run_output_dir=run_output_dir,
        )

        # Extract the ideas and return them in the expected response format
        generated_ideas = result.get("generated_ideas", [])
        print(f"Generated {len(generated_ideas)} ideas.")
        return IdeationResponse(generated_ideas=generated_ideas)

    except Exception as e:
        print(f"Error during ideation: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate ideas: {e}")
