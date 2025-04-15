import logging
from typing import Dict, Any, List, Optional
import asyncio

from langchain_core.language_models import BaseLanguageModel

from agents.manim_agent.config import ManimAgentState
from core.log_utils import log_run_details

logger = logging.getLogger(__name__)

DEFAULT_REFLECTION_PROMPT_TEMPLATE = """
You are an expert Manim programmer acting as a reflective assistant within an AI agent loop.
Your goal is to analyze the outcome of the most recent Manim code generation attempt and provide concise, actionable feedback for the *next* attempt.

**Context for Analysis:**

1.  **Initial User Request:**
    ```
    {initial_user_request}
    ```
    
2.  **History of Past Reflections (if any):**
    {past_reflections_summary}

3.  **Current Task Instruction (May be refined from initial request):**
    ```
    {task_instruction}
    ```

4.  **Code Generator's Plan/Thoughts for This Attempt (Attempt #{attempt_index}):**
    ```
    {generation_thoughts}
    ```

5.  **Code Generated This Attempt (Attempt #{attempt_index}):**
    ```python
    {generated_code}
    ```

6.  **Execution/Validation Outcome:**
    - Success: {execution_success}
    - Validation Error (if execution failed): {validation_error}

7.  **Evaluation Outcome:**
    - Rubric Passed: {evaluation_passed}
    - Evaluator Feedback (if evaluation occurred): {evaluation_feedback}

8.  **Evaluation Rubric:**
    ```
    {rubric}
    ```




**Your Task:**

Based *only* on the information provided above, generate a concise reflection. Focus on:
- **Diagnosis:** *Why* did this attempt fail (or succeed but could be improved)? Compare the generator's `thoughts` with the `execution/validation outcome` and `evaluation outcome/feedback`. Did the execution error contradict the plan? Did the evaluation reveal issues the plan didn't foresee? Reference the rubric if evaluation failed. Take into account the `past_reflections_summary` to make sure we're not repeating past mistakes.
- **Prescription:** What *specific*, *actionable* change(s) should the code generator focus on in the **next** attempt to address the diagnosed issues and better meet the `task_instruction` and `rubric`? Avoid vague advice.

**Output Format:**
Provide ONLY the reflection text. Be concise and direct. Do not include greetings, apologies, or summaries of the input. Start directly with the analysis.
Example: "The code failed because the animation duration was too short according to the rubric. Next attempt should increase `run_time` to 3 seconds for the `Transform`."
"""


class Reflector:
    """
    A component responsible for analyzing the outcome of a Manim generation attempt
    and generating reflections to guide subsequent attempts.
    """

    def __init__(self, llm_text_client: BaseLanguageModel):
        """
        Initializes the Reflector with a text generation LLM client.

        Args:
            llm_text_client: The pre-initialized Langchain LLM client for text generation.
        """
        if not llm_text_client:
            raise ValueError("LLM client cannot be None for Reflector")
        self.llm_text_client = llm_text_client
        logger.info(f"Reflector initialized with LLM: {llm_text_client.__class__.__name__}")

    def _format_past_reflections(self, reflection_history: List[Dict[str, Any]]) -> str:
        """Formats past reflections for inclusion in the prompt."""
        if not reflection_history:
            return "None"
        formatted = []
        for entry in reflection_history:
            idx = entry.get("attempt_index", "?")
            reflection = entry.get("reflection", "[Missing Reflection]")
            formatted.append(f"  - Reflection after Attempt #{idx}: {reflection}")
        return "\\n".join(formatted)

    async def reflect_on_attempt(self, state: ManimAgentState) -> Dict[str, Any]:
        """
        Analyzes the latest attempt and generates a reflection.

        Args:
            state: The current ManimAgentState.

        Returns:
            A dictionary containing the updated 'reflection_history'.
            Includes 'error_message' if reflection generation fails.
        """
        logger.info("Starting reflection on the latest attempt...")
        node_name = "Reflector"

        # --- Extract necessary data from state ---
        initial_user_request = state.get("initial_user_request", "[Not Provided]")
        task_instruction = state.get("task_instruction", "[Not Provided]")
        generation_history = state.get("generation_history", [])
        reflection_history = state.get("reflection_history", [])
        execution_success = state.get("execution_success")
        validation_error = state.get("validation_error", "None")
        evaluation_passed = state.get("evaluation_passed")
        evaluation_result = state.get("evaluation_result", {})
        evaluation_feedback = evaluation_result.get(
            "feedback", "None (Evaluation might not have run or passed)"
        )
        rubric = state.get("rubric", "[Not Provided]")
        attempt_number = state.get("attempt_number", 0)  # Number of *previous* attempts

        # Get details from the *latest* generation attempt
        latest_generation_entry = generation_history[-1] if generation_history else {}
        current_attempt_index = latest_generation_entry.get(
            "attempt_index", attempt_number + 1
        )  # Should match attempt+1
        generation_thoughts = latest_generation_entry.get("thoughts", "[Not Available]")
        generated_code = latest_generation_entry.get("code", "[Not Available]")

        # Ensure indices align, log warning if not
        if current_attempt_index != attempt_number + 1:
            logger.warning(
                f"Mismatch between state.attempt_number ({attempt_number}) and latest generation_history index ({current_attempt_index-1}). Using history index."
            )
            # We'll trust the index in the generation history as it relates directly to the thoughts/code being analyzed.

        # --- Past reflections for context ---

        # --- Log Node Entry ---
        run_output_dir = state.get("run_output_dir", ".")  # Extract run_output_dir
        log_run_details(
            run_output_dir=run_output_dir,
            attempt_number=current_attempt_index,
            node_name=node_name,
            log_category="Node Entry",
            content=f"Starting {node_name} - Analyzing attempt #{current_attempt_index}",
        )
        # ----------------------

        past_reflections_summary = self._format_past_reflections(reflection_history)

        # --- Determine which rubric to use (filled or base) ---
        rubric_to_use = rubric  # Default to base rubric

        # If evaluation passed is False and there is evaluation feedback, use that as the filled rubric
        if (
            evaluation_passed is False
            and evaluation_feedback
            and evaluation_feedback != "None (Evaluation might not have run or passed)"
        ):
            logger.info("Using filled rubric from evaluation feedback")
            rubric_to_use = evaluation_feedback
        # If execution failed, use the base rubric
        elif execution_success is False:
            logger.info("Using base rubric since execution failed")
        else:
            logger.info("Using base rubric (default)")

        # --- Prepare Prompt ---
        prompt = DEFAULT_REFLECTION_PROMPT_TEMPLATE.format(
            initial_user_request=initial_user_request,
            task_instruction=task_instruction,
            attempt_index=current_attempt_index,
            generation_thoughts=generation_thoughts,
            generated_code=generated_code,
            execution_success=(
                str(execution_success) if execution_success is not None else "[Not Run/Completed]"
            ),
            validation_error=validation_error if execution_success is False else "None",
            evaluation_passed=(
                str(evaluation_passed) if evaluation_passed is not None else "[Not Run/Completed]"
            ),
            evaluation_feedback=evaluation_feedback,
            rubric=rubric_to_use,
            past_reflections_summary=past_reflections_summary,
        )

        logger.debug(
            f"Reflection prompt created for attempt #{current_attempt_index}:\\n{prompt[:500]}..."
        )

        # --- Log LLM Prompt ---
        log_run_details(
            run_output_dir=run_output_dir,
            attempt_number=current_attempt_index,
            node_name=node_name,
            log_category="LLM Prompt",
            content=f"Reflection prompt for attempt #{current_attempt_index}:\n---\n{prompt}\n---",
        )
        # ----------------------

        # --- Call LLM ---
        try:
            logger.info(
                f"Invoking LLM for reflection (analyzing attempt #{current_attempt_index})..."
            )
            # Add timeout to prevent hanging
            try:
                # Create a task with an explicit timeout
                llm_task = asyncio.create_task(self.llm_text_client.ainvoke(prompt))
                llm_response = await asyncio.wait_for(llm_task, timeout=120)  # 1 minute timeout
                logger.info(f"Reflection LLM response received within timeout")
            except asyncio.TimeoutError:
                logger.error(f"Reflection LLM request timed out after 120 seconds")
                return {
                    "reflection_history": reflection_history,  # Return original history
                    "error_message": "Reflection LLM request timed out after 120 seconds.",
                }

            # Use .content if available (newer LangChain), otherwise assume string
            reflection_text = (
                llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            )
            reflection_text = reflection_text.strip()
            logger.debug(f"LLM raw reflection response received: {reflection_text[:500]}...")

            # --- Log LLM Response --- (Only if successful and non-empty)
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=current_attempt_index,
                node_name=node_name,
                log_category="LLM Response",
                content=f"Attempt #{current_attempt_index} Reflection: {reflection_text}",
            )
            # ------------------------

            if not reflection_text:
                logger.warning("LLM returned an empty reflection.")
                # Don't add an entry, just return original history and indicate error
                return {
                    "reflection_history": reflection_history,  # Return original history
                    "error_message": "LLM returned empty reflection.",
                }
            else:
                error_message = None  # Clear previous error if successful

            # --- Update Reflection History ---
            new_reflection_entry = {
                "attempt_index": current_attempt_index,  # Index of the attempt being reflected upon
                "reflection": reflection_text,
            }
            updated_reflection_history = reflection_history + [
                new_reflection_entry
            ]  # Append new entry

            logger.info(f"Reflection generated successfully for attempt #{current_attempt_index}.")

            # --- Log Node Completion ---
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=current_attempt_index,
                node_name=node_name,
                log_category="Node Completion",
                content=f"Reflection successful for attempt #{current_attempt_index}.",
            )
            # ---------------------------

            return {
                "reflection_history": updated_reflection_history,
                "error_message": error_message,  # Will be None if successful
            }

        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during reflection generation for attempt #{current_attempt_index}: {e}",
                exc_info=True,
            )

            # --- Log Node Error ---
            error_message_detail = f"Reflection failed for attempt #{current_attempt_index}: {e}"
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=current_attempt_index,
                node_name=node_name,
                log_category="Reflector Error",
                content=error_message_detail,
                is_error=True,
            )
            # ----------------------

            # Log the error but DO NOT append a failure message to the history.
            # Return the original history and signal the error.
            return {
                "reflection_history": reflection_history,  # Return original history on error
                "error_message": f"Reflection generation failed: {e}",
            }
