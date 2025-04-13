import logging
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel

from agents.manim_agent.config import ManimAgentState, AGENT_NAME
from core.log_utils import log_run_details

logger = logging.getLogger(__name__)


class RubricModifier:
    """Handles modification of the evaluation rubric based on enhancement requests."""

    NODE_NAME = "RubricModifier"

    def __init__(self, llm_client: BaseLanguageModel):
        """
        Initializes the RubricModifier.

        Args:
            llm_client: The LLM client for generating modified rubrics.
        """
        if not llm_client:
            raise ValueError("LLM client cannot be None for RubricModifier")
        self.llm_client = llm_client
        self.prompt_template = self._build_prompt_template()
        logger.info(f"RubricModifier initialized with LLM: {llm_client.__class__.__name__}")

    def _build_prompt_template(self) -> str:
        """Builds the prompt template for the LLM call."""
        return (
            "You are an AI assistant helping to refine an evaluation rubric for a Manim animation. "
            "The user wants to enhance a previous animation based on a specific request. "
            "Your task is to update the original rubric to accurately reflect the *new* goals described in the enhancement request. "
            "Focus ONLY on incorporating the enhancement criteria. Maintain the original rubric's structure and intent otherwise.\n\n"
            "--- Original Rubric ---\n{original_rubric}\n--- End Original Rubric ---\n\n"
            "--- Enhancement Request ---\n{enhancement_request}\n--- End Enhancement Request ---\n\n"
            "Generate the **complete modified rubric** based on the enhancement request. Do not add explanations or commentary, only the rubric itself."
        )

    async def modify_rubric_for_enhancement(self, state: ManimAgentState) -> Dict[str, Any]:
        """Modifies the rubric if an enhancement is requested on the first attempt."""
        run_output_dir = Path(state["run_output_dir"])
        # Use attempt_number (0-based index of *previous* attempts)
        # This node runs at the start, so attempt_number is 0 for the first real attempt.
        current_attempt_number = state.get("attempt_number", 0)
        log_attempt = current_attempt_number + 1  # For logging purposes (1-based)

        enhancement_request = state.get("enhancement_request")
        original_rubric = state.get("initial_rubric")  # Use initial_rubric as the base

        log_run_details(
            run_output_dir,
            log_attempt,
            self.NODE_NAME,
            "Node Entry",
            f"Starting {self.NODE_NAME}...",
        )

        # Only modify if it's the *first* attempt (attempt_number is 0) and an enhancement request exists
        if current_attempt_number == 0 and enhancement_request and original_rubric:
            logger.info(
                f"Attempt {log_attempt}: Enhancement request found, attempting to modify rubric."
            )
            log_run_details(
                run_output_dir,
                log_attempt,
                self.NODE_NAME,
                "Action",
                "Enhancement request found on first attempt. Modifying rubric.",
            )

            prompt = self.prompt_template.format(
                original_rubric=original_rubric, enhancement_request=enhancement_request
            )
            log_run_details(run_output_dir, log_attempt, self.NODE_NAME, "LLM Prompt", prompt)

            try:
                logger.info(f"Calling Rubric Modifier LLM: {self.llm_client.__class__.__name__}")
                response = await self.llm_client.ainvoke(prompt)
                modified_rubric = (
                    response.content if hasattr(response, "content") else str(response)
                )
                modified_rubric = modified_rubric.strip()

                log_run_details(
                    run_output_dir,
                    log_attempt,
                    self.NODE_NAME,
                    "LLM Response",
                    modified_rubric,
                )

                if modified_rubric:
                    logger.info("Successfully generated modified rubric.")
                    log_run_details(
                        run_output_dir,
                        log_attempt,
                        self.NODE_NAME,
                        "Node Completion",
                        "Rubric modified successfully.",
                    )
                    return {"rubric": modified_rubric}  # Return only the updated rubric
                else:
                    logger.warning(
                        "Rubric Modifier LLM returned empty content. Using original rubric."
                    )
                    log_run_details(
                        run_output_dir,
                        log_attempt,
                        self.NODE_NAME,
                        "Warning",
                        "LLM returned empty content. Using original rubric.",
                        is_error=True,
                    )
                    # Fall through to return original rubric

            except Exception as e:
                error_message = f"Error during rubric modification LLM call: {e}"
                logger.error(error_message, exc_info=True)
                log_run_details(
                    run_output_dir,
                    log_attempt,
                    self.NODE_NAME,
                    "LLM Error",
                    error_message,
                    is_error=True,
                )
                # Fall through to return original rubric on error

        # If not first attempt, no enhancement request, no rubric, or modification failed:
        # Use the rubric already in the state (which might be the initial or a previously modified one)
        current_rubric = state.get("rubric", original_rubric)  # Fallback needed?
        if current_attempt_number > 0:
            log_message = "Not the first attempt, skipping rubric modification."
        elif not enhancement_request:
            log_message = "No enhancement request, skipping rubric modification."
        else:  # Handle cases where modification failed or original_rubric was None
            log_message = "Rubric modification skipped (no request/rubric, not first attempt, or modification failed). Using current/original rubric."

        logger.info(log_message)
        log_run_details(run_output_dir, log_attempt, self.NODE_NAME, "Info", log_message)
        log_run_details(
            run_output_dir,
            log_attempt,
            self.NODE_NAME,
            "Node Completion",
            "Rubric modification skipped or failed. Using existing rubric.",
        )

        # Return the current rubric (could be initial or unmodified)
        # Ensure we always return *something* for the rubric key if expected by the graph.
        return {"rubric": current_rubric}
