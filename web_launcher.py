import sys
import os
import importlib
from typing import Optional

import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env file from project root before anything else
dotenv.load_dotenv(dotenv.find_dotenv())

# Initialize FastAPI app
app = FastAPI()

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
    script_context: Optional[str] = None
    metadata: Optional[str] = None
    save_generated_code: bool = False


# Define API endpoint
@app.post("/run/{agent_name}")
async def run_agent(agent_name: str, request_data: AgentRunRequest):
    """
    Runs the specified agent by directly importing and calling its execute function.
    """
    print(f"--- Received request for agent: {agent_name} ---")
    runner_module_path = f"agents.{agent_name}.runner"

    # --- Debugging Path Info ---
    print(f"DEBUG (web_launcher): os.getcwd() = {os.getcwd()}")
    print(f"DEBUG (web_launcher): sys.path = {sys.path}")
    # Explicitly add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        print(f"DEBUG (web_launcher): Adding {project_root} to sys.path")
        sys.path.insert(0, project_root)
    # --- End Debugging ---

    try:
        # --- Test Import Parent Package ---
        parent_package_path = ".".join(
            runner_module_path.split(".")[:-1]
        )  # e.g., agents.manim_agent
        if parent_package_path:
            print(
                f"DEBUG (web_launcher): Attempting to import parent '{parent_package_path}' first..."
            )
            importlib.import_module(parent_package_path)
            print(f"DEBUG (web_launcher): Successfully imported parent '{parent_package_path}'.")
        # --- End Test ---

        # Dynamically import the specific agent runner module
        print(f"DEBUG (web_launcher): Attempting to import '{runner_module_path}'...")
        runner_module = importlib.import_module(runner_module_path)
        print(f"DEBUG (web_launcher): Successfully imported '{runner_module_path}'.")

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
        # Await the async execute function
        result_dict = await runner_module.execute(
            script_segment=request_data.script_segment,
            script_context=request_data.script_context,
            input_metadata=request_data.metadata,
            save_generated_code=request_data.save_generated_code,
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
    from config import base_config  # Import to get BASE_OUTPUT_DIR

    # Construct the exclude pattern dynamically based on config
    # This assumes BASE_OUTPUT_DIR is relative to project root
    output_dir_name = base_config.BASE_OUTPUT_DIR.name
    reload_excludes_pattern = f"{output_dir_name}/*"
    print(f"Uvicorn reload excluding: {reload_excludes_pattern}")

    # Run the app using uvicorn
    uvicorn.run(
        "web_launcher:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=[reload_excludes_pattern],  # Add exclusion
    )
