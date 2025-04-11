import os
import re
import base64
import mimetypes
import traceback
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from agents.manim_agent import config as agent_cfg
from core.graph_state import GraphState
from core.log_utils import log_run_details


# --- Pydantic Models for Structured Output ---
class EvaluationResult(BaseModel):
    passed: bool = Field(description="Whether the video passed all rubric criteria.")
    feedback: str = Field(
        description="Detailed feedback explaining the pass/fail decision based on the rubric."
    )


# --- Prompt Template ---
EVALUATION_PROMPT_TEMPLATE = """
Evaluate the provided video based on the following criteria (rubric) and the original script segment it visualizes.

Rubric:
{evaluation_rubric}

Original Script Segment:
{input_text}

Video Analysis:
[The video content is provided as a separate multimodal input]

Provide detailed feedback addressing how well the video meets each point in the rubric. Conclude your evaluation with a JSON object matching the following format:

```json
{{
    "passed": boolean, // True if the video adequately meets the rubric, False otherwise
    "feedback": string // Detailed feedback explaining the pass/fail decision
}}
```
"""


class ManimVideoEvaluator:
    """Evaluates the generated Manim video based on criteria using a multimodal LLM."""

    def __init__(self, llm_eval_client: ChatGoogleGenerativeAI):
        """Initializes the evaluator with a multimodal LLM client."""
        self.llm_eval_client = llm_eval_client
        self.parser = JsonOutputParser(pydantic_object=EvaluationResult)
        # Prompt template is used separately for the text part of the multimodal message
        self.prompt_template = PromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)

    def _prepare_video_payload(self, video_path: str) -> Optional[Dict]:
        """Encodes the video file for the LLM payload."""
        try:
            print(f"Preparing video payload for: {video_path}")
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()

            video_base64 = base64.b64encode(video_bytes).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(video_path)
            if not mime_type:
                mime_type = "video/mp4"  # Sensible default
                print(
                    f"WARN: Could not guess MIME type for {video_path}, defaulting to {mime_type}"
                )

            # Structure required by Langchain for multimodal input
            return {
                "type": "image_url",  # Langchain uses "image_url" for multimodal data
                "image_url": {"url": f"data:{mime_type};base64,{video_base64}"},
            }
        except OSError as e:
            print(f"ERROR: OS error preparing video payload for {video_path}: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Unexpected error preparing video payload for {video_path}: {e}")
            return None

    def _parse_evaluation_response(self, llm_output: str) -> Tuple[bool, str]:
        """Parses the LLM response to extract pass/fail and feedback."""
        try:
            parsed_result = self.parser.parse(llm_output)
            return parsed_result.passed, parsed_result.feedback
        except Exception as e:
            print(f"Error parsing evaluation JSON: {e}")
            # Fallback or re-raise depending on desired robustness
            raise  # Re-raise to be caught by the main evaluation logic

    def evaluate_manim_video(self, state: GraphState) -> Dict:
        """Evaluates the Manim video using a multimodal LLM."""
        node_name = "VideoEvaluator"
        run_output_dir = Path(state["run_output_dir"])
        iteration = state["iteration"]
        log_run_details(
            run_output_dir, iteration, node_name, "Node Entry", f"Starting {node_name}..."
        )

        evaluation_history = state.get("evaluation_history", [])[:]
        error_history = state.get("error_history", [])[:]
        validated_artifact_path_relative_to_run = state.get("validated_artifact_path")
        evaluation_rubric = state.get("evaluation_rubric")
        input_text = state.get("input_text")

        updates_to_state: Dict = {
            "evaluation_feedback": None,
            "evaluation_passed": None,
            "evaluation_history": evaluation_history,
            "error_history": error_history,
            "infrastructure_error": None,  # Reset
        }

        # --- Pre-checks ---
        if not validated_artifact_path_relative_to_run:
            error_message = "Evaluation skipped: No validated video artifact path found."
            log_run_details(
                run_output_dir, iteration, node_name, "Input Error", error_message, is_error=True
            )
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(f"Iter {iteration}: EVAL SKIPPED (No Path): {error_message}")
            return updates_to_state

        if not evaluation_rubric:
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "Skipped",
                "No evaluation rubric provided. Skipping evaluation.",
            )
            updates_to_state["evaluation_passed"] = True
            updates_to_state["evaluation_feedback"] = "Evaluation skipped: No rubric provided."
            evaluation_history.append(f"Iter {iteration}: Skipped (No Rubric)")
            return updates_to_state

        full_video_path = run_output_dir / validated_artifact_path_relative_to_run
        if not full_video_path.exists():
            error_message = (
                f"Evaluation skipped: Video file not found at expected path: {full_video_path}."
            )
            log_run_details(
                run_output_dir, iteration, node_name, "Input Error", error_message, is_error=True
            )
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(f"Iter {iteration}: EVAL SKIPPED (File Missing): {error_message}")
            return updates_to_state

        # --- Video Size Check (Optional, keep for now) ---
        try:
            limit_mb = agent_cfg.EVALUATION_VIDEO_SIZE_LIMIT_MB
            size_bytes = full_video_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            if size_mb > limit_mb:
                error_message = (
                    f"Evaluation skipped: Video file too large ({size_mb:.2f} MB > {limit_mb} MB)."
                )
                log_run_details(
                    run_output_dir,
                    iteration,
                    node_name,
                    "Input Error",
                    error_message,
                    is_error=True,
                )
                updates_to_state["evaluation_feedback"] = error_message
                updates_to_state["evaluation_passed"] = False
                error_history.append(f"Iter {iteration}: EVAL SKIPPED (Too Large): {error_message}")
                return updates_to_state
        except OSError as e:
            error_message = (
                f"Evaluation skipped: Error checking video file size for {full_video_path}: {e}"
            )
            log_run_details(
                run_output_dir, iteration, node_name, "Input Error", error_message, is_error=True
            )
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(
                f"Iter {iteration}: EVAL SKIPPED (Size Check Failed): {error_message}"
            )
            return updates_to_state

        # --- Prepare Prompt and Payload ---
        log_run_details(
            run_output_dir,
            iteration,
            node_name,
            "Prepare Payload",
            f"Preparing video payload for: {full_video_path}",
        )
        video_payload = self._prepare_video_payload(str(full_video_path))
        if not video_payload:
            error_message = (
                f"Evaluation skipped: Failed to prepare video payload for {full_video_path}."
            )
            log_run_details(
                run_output_dir, iteration, node_name, "Input Error", error_message, is_error=True
            )
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(
                f"Iter {iteration}: EVAL SKIPPED (Payload Failed): {error_message}"
            )
            return updates_to_state

        prompt_text = self.prompt_template.format(
            evaluation_rubric=evaluation_rubric, input_text=input_text
        )
        message = HumanMessage(content=[prompt_text, video_payload])
        log_run_details(run_output_dir, iteration, node_name, "Evaluation Prompt Text", prompt_text)
        # Avoid logging the full video payload here, log path instead
        log_run_details(
            run_output_dir, iteration, node_name, "Evaluation Video Path", str(full_video_path)
        )

        # --- Call LLM ---
        llm_output = None
        try:
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "LLM Call",
                f"Calling Evaluation LLM: {self.llm_eval_client.model}",
            )
            response = self.llm_eval_client.invoke([message])
            llm_output = response.content
            log_run_details(
                run_output_dir, iteration, node_name, "Raw Evaluation Output", llm_output
            )

            # --- Parse Response --- (Now uses JsonOutputParser)
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "Parse Result",
                "Attempting to parse JSON output.",
            )
            # Use the parser method directly if stand-alone parsing needed, but error handling below covers it
            # parsed_result = self.parser.parse(llm_output)
            # evaluation_passed, feedback = parsed_result.passed, parsed_result.feedback
            evaluation_passed, feedback = self._parse_evaluation_response(llm_output)

            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "Parsed Result",
                f"Passed: {evaluation_passed}, Feedback: {feedback}",
            )
            updates_to_state["evaluation_passed"] = evaluation_passed
            updates_to_state["evaluation_feedback"] = feedback
            evaluation_history.append(
                f"Iter {iteration}: {'Passed' if evaluation_passed else 'Failed'} - {feedback}"
            )

        except Exception as e:
            error_message = f"Error during evaluation LLM call or parsing: {e}"
            tb_str = traceback.format_exc()
            full_error_details = (
                f"{error_message}\nLLM Output (if available):\n{llm_output}\nTraceback:\n{tb_str}"
            )
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "LLM/Parsing Error",
                full_error_details,
                is_error=True,
            )
            # Treat evaluation failures as potential infrastructure errors or LLM flakiness
            updates_to_state["infrastructure_error"] = error_message
            updates_to_state["evaluation_passed"] = False  # Explicitly fail
            updates_to_state["evaluation_feedback"] = f"Evaluation failed due to error: {e}"
            error_history.append(
                f"Iter {iteration}: EVAL INFRA ERROR: {error_message}"
            )  # Keep error history lean
            evaluation_history.append(f"Iter {iteration}: FAILED (Error: {e})")
            return updates_to_state

        log_run_details(
            run_output_dir,
            iteration,
            node_name,
            "Node Completion",
            f"Finished {node_name}. Updates: { {k:v for k,v in updates_to_state.items() if k not in ['error_history', 'evaluation_history']} }",
        )
        return updates_to_state
