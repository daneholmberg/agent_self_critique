import sys
import os
import importlib
from typing import Optional

import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the new router
from api import config_routes as config_api

# Load .env file from project root before anything else
dotenv.load_dotenv(dotenv.find_dotenv())

# Initialize FastAPI app
app = FastAPI()

# Include the config API router
app.include_router(config_api.router)

# Enable CORS
origins = [
    "http://localhost:3000",  # Typical React dev port
    "http://localhost:5173",  # Typical Vite/React dev port
    "http://localhost:5174",  # Add the port currently in use
    # Add any other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    # allow_origins=["*"], # Or allow all origins for local dev simplicity
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define request data model
class AgentRunRequest(BaseModel):
    script_segment: str
    general_context: Optional[str] = None
    previous_code_attempt: Optional[str] = None
    enhancement_request: Optional[str] = None
    final_command: Optional[str] = None
    scene_name: str
    save_generated_code: bool = False


# Define API endpoint
@app.post("/run/{agent_name}")
async def run_agent(agent_name: str, request_data: AgentRunRequest):
    """
    Runs the specified agent by directly importing and calling its execute function.
    """
    print(f"--- Received request for agent: {agent_name} ---")
    runner_module_path = f"agents.{agent_name}.runner"

    # Explicitly add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:

        sys.path.insert(0, project_root)
    # --- End Debugging ---

    try:
        # --- Test Import Parent Package ---
        parent_package_path = ".".join(
            runner_module_path.split(".")[:-1]
        )  # e.g., agents.manim_agent
        if parent_package_path:

            importlib.import_module(parent_package_path)
        # --- End Test ---

        # Dynamically import the specific agent runner module
        runner_module = importlib.import_module(runner_module_path)

    except ModuleNotFoundError as e:
        print(f"Error during import: {e}")  # Print the specific error
        print(f"Error: Runner module import failed for path '{runner_module_path}'.")
        raise HTTPException(
            status_code=404,
            detail=f"Agent runner import failed for '{runner_module_path}'. Error: {e}",
        )
    except Exception as e:
        print(f"Error importing runner module '{runner_module_path}': {e}")
        # Include traceback for server logs
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error importing agent runner: {e}")

    # Check if the runner module has an 'execute' function
    if not (hasattr(runner_module, "execute") and callable(runner_module.execute)):
        print(
            f"Error: Runner module '{runner_module_path}' does not have a callable 'execute' function."
        )
        raise HTTPException(
            status_code=501,
            detail=f"Agent runner '{runner_module_path}' has no valid 'execute' function.",
        )

    # Call the agent's execute function
    try:
        print(f"Calling {runner_module_path}.execute...")
        # Await the async execute function with updated arguments
        result_dict = await runner_module.execute(
            script_segment=request_data.script_segment,
            general_context=request_data.general_context,
            previous_code_attempt=request_data.previous_code_attempt,
            enhancement_request=request_data.enhancement_request,
            final_command=request_data.final_command,
            scene_name=request_data.scene_name,
            save_generated_code=request_data.save_generated_code,
            # run_output_dir_override could be added here if needed in request
        )
        print("Agent execution finished.")

        # Return the result dictionary directly (or adapt as needed)
        # We can add more details like stdout/stderr if we capture them
        # during the direct call (e.g., by redirecting sys.stdout/stderr)
        # For now, we return the structured result.
        return result_dict

    except Exception as e:
        # Catch potential errors during the agent execution itself
        print(f"Error during execution of agent '{agent_name}': {e}")
        # Include traceback for server logs
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during agent execution: {e}")


# Basic root endpoint for testing if the server is running
@app.get("/")
async def read_root():
    return {"message": "Manim Automator Agent Runner Backend"}


if __name__ == "__main__":
    import uvicorn
    from config import base_config  # Keep this if needed elsewhere, but not for excludes now

    # Define patterns to explicitly INCLUDE for reloading
    # Watch .py files within these core directories recursively
    include_patterns = [
        "web_launcher.py",
        "agents/**/*.py",
        "core/**/*.py",
        "config/**/*.py",
        "api/**/*.py",
        ".env",  # Reload if environment variables change
    ]

    # Define patterns to EXCLUDE
    # Start by excluding all Python files, then add specific directories like outputs
    exclude_patterns = [
        "**/*.py",  # Exclude ALL .py files first
        "**/__pycache__/*",  # Ignore pycache
        "**/*.pyc",  # Ignore compiled python
        "outputs/**",  # Ignore everything in outputs
        # Add other specific directories/files to ignore if needed
        # e.g., "venv/**", ".git/**"
    ]

    print(f"Uvicorn reload including: {include_patterns}")
    print(f"Uvicorn reload excluding: {exclude_patterns}")

    # Run the app using uvicorn
    uvicorn.run(
        "web_launcher:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_includes=include_patterns,
        reload_excludes=exclude_patterns,
    )
