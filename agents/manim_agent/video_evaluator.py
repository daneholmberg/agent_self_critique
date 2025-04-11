import os
import re
import base64
import traceback
import cv2
import numpy as np
import json
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError

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
Evaluate the provided video frames based on the following criteria (rubric) and the original script segment they visualize. The frames are extracted at 1 frame per second, plus the final frame.

Rubric:
{evaluation_rubric}

Original Script Segment:
{input_text}

Video Analysis:
[The video frame content is provided as separate multimodal inputs]

Provide detailed feedback addressing how well the video meets each point in the rubric based *only* on the provided frames. Conclude your evaluation with a JSON object matching the following format:

```json
{{
    "passed": boolean, // True if the video adequately meets the rubric, False otherwise
    "feedback": string // Detailed feedback explaining the pass/fail decision
}}
```
"""


class ManimVideoEvaluator:
    """Evaluates the generated Manim video based on criteria using a multimodal LLM."""

    NODE_NAME = "VideoEvaluator"  # Class constant for node name

    def __init__(self, llm_eval_client: ChatGoogleGenerativeAI):
        """Initializes the evaluator with a multimodal LLM client."""
        self.llm_eval_client = llm_eval_client
        self.parser = JsonOutputParser(pydantic_object=EvaluationResult)
        self.prompt_template = PromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)

    # --- Frame Extraction Helpers --- #

    def _calculate_frame_indices(
        self,
        fps: float,
        total_frames: int,
        target_fps: int,
        max_frames_to_send: int,
    ) -> List[int]:
        """Calculates the specific frame indices to extract based on configuration."""
        if total_frames <= 0:
            return []

        indices_to_extract = set()
        frame_interval = max(1, int(round(fps / target_fps)))  # Frames to skip

        for i in range(0, total_frames, frame_interval):
            indices_to_extract.add(i)
        indices_to_extract.add(total_frames - 1)  # Ensure last frame

        # Limit number of frames if necessary, sampling evenly
        if len(indices_to_extract) > max_frames_to_send:
            print(
                f"WARN: Initial frame count ({len(indices_to_extract)}) exceeds limit ({max_frames_to_send}). Sampling evenly."
            )
            sampled_indices = np.linspace(0, total_frames - 1, num=max_frames_to_send, dtype=int)
            indices_to_extract = set(sampled_indices)

        sorted_indices = sorted(list(indices_to_extract))
        print(f"Final frame indices to extract ({len(sorted_indices)} frames): {sorted_indices}")
        return sorted_indices

    def _process_selected_frames(
        self, cap: cv2.VideoCapture, sorted_indices: List[int], video_path: str  # For logging
    ) -> List[Dict]:
        """Reads, encodes, and formats the specified frames into payloads."""
        frames_payload = []
        extracted_frame_count = 0
        for frame_index in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                print(f"WARN: Could not read frame at index {frame_index} from {video_path}")
                continue

            # Encode frame as JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                print(f"WARN: Could not encode frame {frame_index} to JPEG.")
                continue

            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # Structure required by Langchain
            frame_payload = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
            }
            frames_payload.append(frame_payload)
            extracted_frame_count += 1

        print(f"Successfully extracted and encoded {extracted_frame_count} frames.")
        return frames_payload

    def _prepare_video_payload(self, video_path: str) -> Optional[List[Dict]]:
        """Opens video, calculates indices, and processes selected frames."""
        target_fps = agent_cfg.VIDEO_EVAL_FRAMES_PER_SECOND
        max_frames_to_send = agent_cfg.VIDEO_EVAL_MAX_FRAMES
        cap = None
        try:
            print(f"Preparing video frames payload for: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"ERROR: Could not open video file: {video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video Info: FPS={fps}, Total Frames={total_frames}")

            if fps <= 0 or total_frames <= 0:
                print(
                    f"ERROR: Invalid video properties (FPS: {fps}, Frames: {total_frames}) for {video_path}"
                )
                return None

            sorted_indices = self._calculate_frame_indices(
                fps, total_frames, target_fps, max_frames_to_send
            )

            if not sorted_indices:
                print("WARN: No valid frame indices calculated.")
                return []  # Return empty list, not None, if no frames selected

            frames_payload = self._process_selected_frames(cap, sorted_indices, video_path)

            return frames_payload
        except cv2.error as e:
            print(f"ERROR: OpenCV error processing video {video_path}: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Unexpected error preparing video frame payload for {video_path}: {e}")
            traceback.print_exc()
            return None
        finally:
            if cap:
                cap.release()

    # --- LLM Interaction Helpers --- #

    def _parse_evaluation_response(self, llm_output: str) -> Tuple[bool, str]:
        """Parses the LLM response to extract pass/fail and feedback."""
        try:
            print(f"--- Attempting to parse LLM output ---")
            print(
                f"Raw LLM Output (type: {type(llm_output)}):\n{repr(llm_output)}"
            )  # Log raw string with repr

            parsed_result = self.parser.parse(llm_output)
            print(f"Parser Output Type: {type(parsed_result)}")
            print(f"Parser Output Value: {parsed_result}")

            # --- Investigation Step: Direct Pydantic Validation ---
            try:
                # Attempt parsing with standard json first
                dict_result = json.loads(
                    llm_output.strip()
                )  # Ensure no leading/trailing whitespace
                print("Successfully parsed with json.loads")
                # Now validate with Pydantic
                validated_model = EvaluationResult.model_validate(dict_result)
                print(f"Successfully validated with Pydantic: {type(validated_model)}")
                # If direct validation works, we use its result
                return validated_model.passed, validated_model.feedback
            except (json.JSONDecodeError, ValidationError, TypeError) as direct_e:
                print(f"Direct JSON/Pydantic validation failed: {direct_e}")
                # If direct validation fails, we fall back to the parser's potentially problematic output
                # This keeps the previous workaround logic as a fallback
                if isinstance(parsed_result, EvaluationResult):
                    print("Falling back to parser's Pydantic object")
                    return parsed_result.passed, parsed_result.feedback
                elif isinstance(parsed_result, dict):
                    print("Falling back to parser's dict object")
                    passed = parsed_result.get("passed")
                    feedback = parsed_result.get("feedback")
                    if passed is None or feedback is None:
                        raise ValueError(
                            "Parsed dictionary missing 'passed' or 'feedback' key (fallback)"
                        )
                    return passed, feedback
                else:
                    # If neither direct validation nor parser output is usable
                    raise TypeError(
                        f"Unexpected parse result type after fallback: {type(parsed_result)}"
                    )

        except Exception as e:
            print(f"Error parsing evaluation JSON: {e}")
            # Fallback or re-raise depending on desired robustness
            raise  # Re-raise to be caught by the main evaluation logic

    def _invoke_and_parse_llm(
        self,
        message: HumanMessage,
        run_output_dir: Path,
        iteration: int,
    ) -> Tuple[Optional[Tuple[bool, str]], Optional[str]]:
        """Invokes the LLM, parses the response, and handles errors.

        Returns:
            Tuple containing:
                - (passed, feedback) tuple if successful, else None.
                - error_message string if an error occurred, else None.
        """
        llm_output = None
        try:
            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "LLM Call",
                f"Calling Evaluation LLM: {self.llm_eval_client.model}",
            )
            response = self.llm_eval_client.invoke([message])
            llm_output = response.content
            log_run_details(
                run_output_dir, iteration, self.NODE_NAME, "Raw Evaluation Output", llm_output
            )

            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "Parse Result",
                "Attempting to parse JSON output.",
            )
            evaluation_passed, feedback = self._parse_evaluation_response(llm_output)
            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "Parsed Result",
                f"Passed: {evaluation_passed}, Feedback: {feedback}",
            )
            return (evaluation_passed, feedback), None  # Success

        except Exception as e:
            error_message = f"Error during evaluation LLM call or parsing: {e}"
            tb_str = traceback.format_exc()
            full_error_details = (
                f"{error_message}\nLLM Output (if available):\n{llm_output}\nTraceback:\n{tb_str}"
            )
            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "LLM/Parsing Error",
                full_error_details,
                is_error=True,
            )
            return None, error_message  # Failure

    # --- Main Orchestration Logic --- #

    def _perform_pre_checks(
        self,
        state: GraphState,
        run_output_dir: Path,
        iteration: int,
    ) -> Tuple[bool, Optional[Dict], Optional[Path], Optional[str], Optional[str]]:
        """Performs initial checks on state and files.

        Returns:
            Tuple containing:
                - bool: True if checks passed, False otherwise.
                - Optional[Dict]: State updates if checks failed, else None.
                - Optional[Path]: Full path to the video if checks passed, else None.
                - Optional[str]: Evaluation rubric if checks passed, else None.
                - Optional[str]: Input text if checks passed, else None.
        """
        evaluation_history = state.get("evaluation_history", [])
        error_history = state.get("error_history", [])
        validated_artifact_path_relative_to_run = state.get("validated_artifact_path")
        evaluation_rubric = state.get("rubric")
        input_text = state.get("input_text")

        updates_to_state: Dict = {
            "evaluation_feedback": None,
            "evaluation_passed": None,
            "evaluation_history": evaluation_history[:],  # Copy
            "error_history": error_history[:],  # Copy
            "infrastructure_error": None,
        }

        if not validated_artifact_path_relative_to_run:
            error_message = "Evaluation skipped: No validated video artifact path found."
            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "Input Error",
                error_message,
                is_error=True,
            )
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            updates_to_state["error_history"].append(
                f"Iter {iteration}: EVAL SKIPPED (No Path): {error_message}"
            )
            return False, updates_to_state, None, None, None

        if not evaluation_rubric:
            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "Skipped",
                "No evaluation rubric provided. Skipping evaluation.",
            )
            updates_to_state["evaluation_passed"] = True
            updates_to_state["evaluation_feedback"] = "Evaluation skipped: No rubric provided."
            updates_to_state["evaluation_history"].append(f"Iter {iteration}: Skipped (No Rubric)")
            return False, updates_to_state, None, None, None

        full_video_path = run_output_dir / validated_artifact_path_relative_to_run
        if not full_video_path.exists():
            error_message = (
                f"Evaluation skipped: Video file not found at expected path: {full_video_path}."
            )
            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "Input Error",
                error_message,
                is_error=True,
            )
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            updates_to_state["error_history"].append(
                f"Iter {iteration}: EVAL SKIPPED (File Missing): {error_message}"
            )
            return False, updates_to_state, None, None, None

        # If all checks pass
        return True, None, full_video_path, evaluation_rubric, input_text

    def evaluate_manim_video(self, state: GraphState) -> Dict:
        """Evaluates the Manim video using frame extraction and a multimodal LLM."""
        run_output_dir = Path(state["run_output_dir"])
        iteration = state["iteration"]
        log_run_details(
            run_output_dir, iteration, self.NODE_NAME, "Node Entry", f"Starting {self.NODE_NAME}..."
        )

        # Perform pre-checks
        checks_passed, failure_updates, full_video_path, evaluation_rubric, input_text = (
            self._perform_pre_checks(state, run_output_dir, iteration)
        )
        if not checks_passed:
            return failure_updates  # Return early if checks failed

        # Initialize state update dictionary for this run
        evaluation_history = state.get("evaluation_history", [])[:]
        error_history = state.get("error_history", [])[:]

        updates_to_state: Dict = {
            "evaluation_feedback": None,
            "evaluation_passed": None,
            "evaluation_history": evaluation_history,
            "error_history": error_history,
            "infrastructure_error": None,  # Reset
        }

        # --- Prepare Prompt and Payload ---
        log_run_details(
            run_output_dir,
            iteration,
            self.NODE_NAME,
            "Prepare Payload",
            f"Extracting frames from video: {full_video_path}",
        )
        # Now returns a list of frame payloads or None
        frame_payloads = self._prepare_video_payload(str(full_video_path))

        # Check if frame extraction failed or returned no frames
        if frame_payloads is None:
            error_message = (
                f"Evaluation skipped: Failed to extract frames from video {full_video_path}."
            )
            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "Input Error",
                error_message,
                is_error=True,
            )
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(
                f"Iter {iteration}: EVAL SKIPPED (Frame Extraction Failed): {error_message}"
            )
            return updates_to_state

        if not frame_payloads:  # Empty list means no frames selected/found
            error_message = f"Evaluation skipped: No frames extracted from video {full_video_path}."
            log_run_details(
                run_output_dir,
                iteration,
                self.NODE_NAME,
                "Input Error",
                error_message,
                is_error=True,
            )
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(
                f"Iter {iteration}: EVAL SKIPPED (No Frames Extracted): {error_message}"
            )
            return updates_to_state

        prompt_text = self.prompt_template.format(
            evaluation_rubric=evaluation_rubric, input_text=input_text
        )
        # Construct message with text prompt followed by all frame payloads
        message_content = [prompt_text] + frame_payloads
        message = HumanMessage(content=message_content)

        log_run_details(
            run_output_dir, iteration, self.NODE_NAME, "Evaluation Prompt Text", prompt_text
        )
        # Avoid logging the full frame payloads, log count and path instead
        log_run_details(
            run_output_dir,
            iteration,
            self.NODE_NAME,
            "Evaluation Frames Info",
            f"Using {len(frame_payloads)} frames from {full_video_path}",
        )

        # --- Call LLM ---
        parsed_result, error_message = self._invoke_and_parse_llm(
            message, run_output_dir, iteration
        )

        if parsed_result:
            evaluation_passed, feedback = parsed_result
            updates_to_state["evaluation_passed"] = evaluation_passed
            updates_to_state["evaluation_feedback"] = feedback
            evaluation_history.append(
                f"Iter {iteration}: {'Passed' if evaluation_passed else 'Failed'} - {feedback}"
            )
        else:  # Error occurred during LLM call/parsing
            # Treat evaluation failures as potential infrastructure errors or LLM flakiness
            updates_to_state["infrastructure_error"] = error_message
            updates_to_state["evaluation_passed"] = False  # Explicitly fail
            updates_to_state["evaluation_feedback"] = (
                f"Evaluation failed due to error: {error_message}"
            )
            error_history.append(
                f"Iter {iteration}: EVAL INFRA ERROR: {error_message}"
            )  # Keep error history lean
            evaluation_history.append(f"Iter {iteration}: FAILED (Error: {error_message})")

        log_run_details(
            run_output_dir,
            iteration,
            self.NODE_NAME,
            "Node Completion",
            f"Finished {self.NODE_NAME}. Updates: { {k:v for k,v in updates_to_state.items() if k not in ['error_history', 'evaluation_history']} }",
        )
        return updates_to_state
