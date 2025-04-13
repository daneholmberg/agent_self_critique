import logging
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from agents.manim_agent.config import ManimAgentState, AGENT_NAME
from core.log_utils import log_run_details

logger = logging.getLogger(__name__)


class RubricModifier:
    """Handles modification of the evaluation rubric based on the user request and context."""

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

    def _build_prompt_template(self) -> PromptTemplate:
        """Builds the prompt template for the LLM call."""
        template_string = (
            "You are an AI assistant helping to refine an evaluation rubric for a Manim animation scene.\n\n"
            "The user wants to create a scene based on the following request:\n"
            "--- Original Base Rubric ---\n{original_rubric}\n--- End Original Base Rubric ---\n\n"
            "--- User Request ---\n{user_request}\n--- End User Request ---\n\n"
            "{previous_code_section}"
            "{enhancement_section}"
            "Your task is to take the original base rubric provided below and adapt it to accurately reflect the specific goals described in the user's request (and enhancement, if provided). You can change the current sections of the rubric to better fit the request, or add new sections if needed."
            "Often you will need to do both, but not always for simple requests."
            "Focus ONLY on incorporating the specific criteria needed to judge the success of *this particular request*. Maintain the original rubric's structure and intent otherwise.\n\n"
            "Generate the **complete modified rubric** tailored to the request. Do not add explanations or commentary, only the rubric itself."
        )
        return PromptTemplate(
            template=template_string,
            input_variables=[
                "user_request",
                "original_rubric",
                "enhancement_section",
                "previous_code_section",
            ],
        )

    async def modify_rubric(self, state: ManimAgentState) -> Dict[str, Any]:
        """Modifies the rubric based on the user request and optional enhancement/previous code."""
        run_output_dir = Path(state["run_output_dir"])
        current_attempt_number = state.get("attempt_number", 0)
        log_attempt = current_attempt_number + 1

        user_request = state.get("initial_user_request")
        enhancement_request = state.get("enhancement_request")
        previous_code = state.get("previous_code_attempt")
        original_rubric = state.get("initial_rubric")

        enhancement_section = ""
        if enhancement_request:
            enhancement_section = (
                f"Additionally, the user provided the following enhancement request to refine the goal:\n"
                f"--- Enhancement Request ---\n{enhancement_request}\n--- End Enhancement Request ---\n\n"
            )

        previous_code_section = ""
        if previous_code and enhancement_request:
            previous_code_section = (
                f"For context, here is the previous code attempt that the enhancement request refers to:\n"
                f"--- Previous Code Attempt ---\n{previous_code}\n--- End Previous Code Attempt ---\n\n"
            )

        log_run_details(
            run_output_dir,
            log_attempt,
            self.NODE_NAME,
            "Node Entry",
            f"Starting {self.NODE_NAME}...",
        )

        if user_request and original_rubric:
            logger.info(f"Attempt {log_attempt}: Preparing to modify rubric based on user request.")
            log_run_details(
                run_output_dir,
                log_attempt,
                self.NODE_NAME,
                "Action",
                "User request and initial rubric found. Modifying rubric.",
            )

            try:
                prompt = self.prompt_template.format(
                    user_request=user_request,
                    original_rubric=original_rubric,
                    enhancement_section=enhancement_section,
                    previous_code_section=previous_code_section,
                )
                # --- Log LLM Prompt --- Start
                prompt_log_path = run_output_dir / f"rubric_modifier_prompt_iter_{log_attempt}.txt"
                try:
                    with open(prompt_log_path, "w", encoding="utf-8") as f:
                        f.write(prompt)
                    logger.info(f"Rubric modifier prompt saved to: {prompt_log_path}")
                except Exception as log_e:
                    logger.warning(f"Could not save rubric modifier prompt log: {log_e}")
                log_run_details(run_output_dir, log_attempt, self.NODE_NAME, "LLM Prompt", prompt)
                # --- Log LLM Prompt --- End

                logger.info(f"Calling Rubric Modifier LLM: {self.llm_client.__class__.__name__}")
                response = await self.llm_client.ainvoke(prompt)
                llm_response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # --- Log LLM Response --- Start
                response_log_path = (
                    run_output_dir / f"rubric_modifier_response_iter_{log_attempt}.txt"
                )
                try:
                    with open(response_log_path, "w", encoding="utf-8") as f:
                        f.write(llm_response_text)
                    logger.info(f"Rubric modifier response saved to: {response_log_path}")
                except Exception as log_e:
                    logger.warning(f"Could not save rubric modifier response log: {log_e}")
                log_run_details(
                    run_output_dir,
                    log_attempt,
                    self.NODE_NAME,
                    "LLM Response",
                    llm_response_text,  # Log the full response text
                )
                # --- Log LLM Response --- End

                modified_rubric = llm_response_text.strip()

                if modified_rubric:
                    logger.info(
                        f"Successfully generated modified rubric for attempt {log_attempt}."
                    )
                    log_run_details(
                        run_output_dir,
                        log_attempt,
                        self.NODE_NAME,
                        "Node Completion",
                        "Rubric modified successfully.",
                    )
                    return {"rubric": modified_rubric}
                else:
                    logger.warning(
                        f"Attempt {log_attempt}: Rubric Modifier LLM returned empty content. Using original rubric."
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
                error_message = (
                    f"Attempt {log_attempt}: Error during rubric modification LLM call: {e}"
                )
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

        else:
            missing_items = []
            if not user_request:
                missing_items.append("initial_user_request")
            if not original_rubric:
                missing_items.append("initial_rubric")
            log_message = f"Skipping rubric modification because required inputs are missing: {', '.join(missing_items)}. Using original rubric."
            logger.warning(log_message)
            log_run_details(
                run_output_dir, log_attempt, self.NODE_NAME, "Warning", log_message, is_error=True
            )

        log_message = "Rubric modification skipped or failed. Using original rubric."
        logger.info(log_message)
        log_run_details(run_output_dir, log_attempt, self.NODE_NAME, "Info", log_message)
        log_run_details(
            run_output_dir,
            log_attempt,
            self.NODE_NAME,
            "Node Completion",
            "Rubric modification skipped or failed. Using original rubric.",
        )

        return {"rubric": original_rubric}
