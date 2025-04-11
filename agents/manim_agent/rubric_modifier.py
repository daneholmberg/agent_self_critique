from typing import Dict, Optional
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from core.graph_state import GraphState
from core.log_utils import log_run_details


class RubricModifier:
    """Handles dynamically modifying the evaluation rubric based on enhancement requests."""

    def __init__(self, llm_client: ChatGoogleGenerativeAI):
        """
        Initializes the rubric modifier with an LLM client.

        Args:
            llm_client: The pre-initialized Langchain LLM client.
        """
        self.llm_client = llm_client

    def _build_rubric_modification_prompt(
        self, original_rubric: str, enhancement_request: str
    ) -> str:
        """
        Constructs the prompt to ask the LLM to add an enhancement criterion to the rubric.
        """
        return f"""You are tasked with updating an evaluation rubric for Manim video generation.

Here is the original rubric:
--- START ORIGINAL RUBRIC ---
{original_rubric}
--- END ORIGINAL RUBRIC ---

A specific enhancement has been requested for the Manim code/video being generated:
--- START ENHANCEMENT REQUEST ---
{enhancement_request}
--- END ENHANCEMENT REQUEST ---

Please add one OR MORE new criterion to the rubric *specifically* related to fulfilling this enhancement request. This new criterion should be phrased clearly, you are allowed to be a little bit more verbose than the other sections to the rubric to make sure there's enough details that the grader can easily understand the new criterion. It must require a score of 4 (Good) or 5 (Excellent) on the standard 1-5 scale for the enhancement to be considered successfully implemented.

Output ONLY the *complete*, updated rubric including the original criteria and the new enhancement criterion. Do not add any introductory text, explanations, or markdown formatting around the rubric itself.
"""

    def modify_rubric_for_enhancement(self, state: GraphState) -> Dict:
        """Adds a criterion to the rubric based on the enhancement_request, if present."""
        node_name = "RubricModifier"
        run_output_dir = Path(state["run_output_dir"])
        # Use current iteration from state, node runs *before* generator increments it
        current_iteration_number = state["iteration"]

        log_run_details(
            run_output_dir,
            current_iteration_number + 1,
            node_name,
            "Node Entry",
            f"Checking for rubric modification...",
        )

        original_rubric = state.get("rubric")
        enhancement_request = state.get("enhancement_request")

        # Only modify if it's the *first* iteration and an enhancement request exists
        if current_iteration_number == 0 and enhancement_request and original_rubric:
            prompt = self._build_rubric_modification_prompt(original_rubric, enhancement_request)
            log_run_details(
                run_output_dir,
                current_iteration_number + 1,
                node_name,
                "LLM Prompt (Rubric Mod)",
                prompt,
            )

            try:
                print(f"Calling LLM to modify rubric: {self.llm_client.model}")
                response = self.llm_client.invoke(prompt)
                modified_rubric = response.content.strip()  # Get the modified rubric text
                log_run_details(
                    run_output_dir,
                    current_iteration_number + 1,
                    node_name,
                    "LLM Response (Rubric Mod)",
                    modified_rubric,
                )

                if modified_rubric:
                    log_run_details(
                        run_output_dir,
                        current_iteration_number + 1,
                        node_name,
                        "Node Completion",
                        "Rubric successfully modified for enhancement.",
                    )
                    return {"rubric": modified_rubric}  # Return update
                else:
                    warning_msg = "LLM returned empty response for rubric modification."
                    print(f"Warning: {warning_msg}")
                    log_run_details(
                        run_output_dir,
                        current_iteration_number + 1,
                        node_name,
                        "LLM Warning",
                        warning_msg,
                    )
                    return {}  # No change

            except Exception as e:
                error_message = f"Gemini API Error during rubric modification: {e}"
                print(f"ERROR: {error_message}")
                log_run_details(
                    run_output_dir,
                    current_iteration_number + 1,
                    node_name,
                    "LLM Error",
                    error_message,
                    is_error=True,
                )
                # Log as infrastructure error so it doesn't stop the main flow if rubric mod fails
                return {"infrastructure_error": f"Rubric modification failed: {error_message}"}

        else:
            # No enhancement request or not the first iteration
            log_run_details(
                run_output_dir,
                current_iteration_number + 1,
                node_name,
                "Node Completion",
                "No rubric modification needed.",
            )
            return {}  # No changes to state
