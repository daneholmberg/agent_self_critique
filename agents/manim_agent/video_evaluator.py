import os
import re
import base64
import traceback
import cv2
import numpy as np
import json
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import logging
import asyncio
import mimetypes
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.language_models import BaseLanguageModel

from agents.manim_agent import config as agent_cfg
from core.graph_state import GraphState
from core.log_utils import log_run_details
from agents.manim_agent.config import (
    ManimAgentState,
    VIDEO_EVAL_MAX_SIZE_MB,
    VIDEO_EVAL_FRAMES_PER_SECOND,
    VIDEO_EVAL_MAX_FRAMES,
)

logger = logging.getLogger(__name__)

# Constants for video processing
MAX_VIDEO_SIZE_BYTES = VIDEO_EVAL_MAX_SIZE_MB * 1024 * 1024


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

Provide detailed feedback addressing how well the video meets each point in the rubric based *only* on the provided frames.
The feedback should be detailed and you should give very rough and interpreable suggestions for improvement.
This is especially true when it's a 4 and not a 5, in the past you won't give any critiques or feedback on 4's.
Don't frame it like that's what they *have* to do, just something to think about.
You can give out half points. So if you don't quite want to give it a 5 but you think it's *very* close, you can give a 4.5.
Remember this is python so true and false are True and False, not true and false.
Remember you only see at roughly 1 frame per second, so you will may only catch the middle of a transition and not the full thing. 
So you need to take this into account when judging the video.
Conclude your evaluation with a JSON object EXACTLY matching the following format:

```json
{{
    "feedback": string // Detailed feedback explaining the pass/fail decision
    "passed": boolean, // True if the video adequately meets the rubric, False otherwise
}}
```
"""


class ManimVideoEvaluator:
    """Evaluates the generated Manim video based on criteria using a multimodal LLM."""

    NODE_NAME = "VideoEvaluator"  # Class constant for node name

    def __init__(self, llm_eval_client: BaseLanguageModel):
        """
        Initializes the video evaluator.

        Args:
            llm_eval_client: A pre-initialized Langchain multimodal LLM client.
        """
        if not llm_eval_client:
            raise ValueError("Multimodal LLM client cannot be None for ManimVideoEvaluator")
        self.llm_eval_client = llm_eval_client
        self.parser = JsonOutputParser(pydantic_object=EvaluationResult)
        self.prompt_template = PromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)
        logger.info(
            f"ManimVideoEvaluator initialized with LLM: {llm_eval_client.__class__.__name__}"
        )

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

    async def _extract_frames(
        self, video_path: str, output_dir: Path, fps: int
    ) -> Optional[List[Path]]:
        """Extracts frames from a video using ffmpeg."""
        logger.info(f"Extracting frames from {video_path} at {fps} FPS...")
        output_pattern = output_dir / "frame_%04d.png"
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            str(output_pattern),
            "-loglevel",  # Add logging level
            "error",  # Only show errors
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            stdout_decoded = stdout.decode().strip()
            stderr_decoded = stderr.decode().strip()

            if process.returncode != 0:
                logger.error(
                    f"ffmpeg failed (code {process.returncode}) to extract frames from {video_path}. Stderr: {stderr_decoded}"
                )
                return None
            else:
                if stderr_decoded:
                    logger.warning(f"ffmpeg stderr (but process succeeded): {stderr_decoded}")
                logger.info(f"ffmpeg stdout: {stdout_decoded if stdout_decoded else '[No stdout]'}")
                # List generated frame files
                extracted_frames = sorted(list(output_dir.glob("frame_*.png")))
                if not extracted_frames:
                    logger.warning(f"ffmpeg ran successfully but no frames found in {output_dir}")
                    return []
                logger.info(
                    f"Successfully extracted {len(extracted_frames)} frames to {output_dir}"
                )
                return extracted_frames
        except FileNotFoundError:
            logger.error("ffmpeg command not found. Ensure ffmpeg is installed and in PATH.")
            return None
        except Exception as e:
            logger.error(f"Error running ffmpeg: {e}", exc_info=True)
            return None

    # --- LLM Interaction Helpers --- #

    # Note: _invoke_and_parse_llm and _parse_evaluation_response are removed
    # Parsing is handled directly in evaluate_manim_video using self.parser

    # --- Main Orchestration Logic --- #

    async def _perform_pre_checks(
        self, state: ManimAgentState, run_output_dir: Path, current_attempt: int
    ) -> Tuple[bool, Dict[str, Any], Optional[str], Optional[str], Optional[str]]:
        """Performs initial checks for required state keys and file existence."""
        updates_on_failure: Dict[str, Any] = {
            "evaluation_result": None,
            "evaluation_passed": False,  # Default to False on pre-check fail
        }

        # Check required state keys
        video_path_str = state.get("video_path")
        evaluation_rubric = state.get("rubric")
        # Use task_instruction as the input text reference
        input_text = state.get("task_instruction")

        if not video_path_str:
            error_message = "Evaluation skipped: 'video_path' missing from state."
            log_run_details(
                run_output_dir,
                current_attempt,
                self.NODE_NAME,
                "Input Error",
                error_message,
                is_error=True,
            )
            updates_on_failure["evaluation_result"] = {"feedback": error_message, "passed": False}
            return False, updates_on_failure, None, None, None

        if not evaluation_rubric:
            error_message = "Evaluation skipped: 'rubric' missing from state."
            log_run_details(
                run_output_dir,
                current_attempt,
                self.NODE_NAME,
                "Input Error",
                error_message,
                is_error=True,
            )
            updates_on_failure["evaluation_result"] = {"feedback": error_message, "passed": False}
            return False, updates_on_failure, None, None, None

        if not input_text:
            # Log warning but proceed, evaluation might still be possible without it
            logger.warning("'task_instruction' missing from state, evaluation context incomplete.")
            log_run_details(
                run_output_dir,
                current_attempt,
                self.NODE_NAME,
                "Input Warning",
                "'task_instruction' missing from state.",
            )

        # Check if video file actually exists
        if not Path(video_path_str).is_file():
            error_message = f"Evaluation skipped: Video file not found at path: {video_path_str}"
            log_run_details(
                run_output_dir,
                current_attempt,
                self.NODE_NAME,
                "File Error",
                error_message,
                is_error=True,
            )
            updates_on_failure["evaluation_result"] = {"feedback": error_message, "passed": False}
            return False, updates_on_failure, None, None, None

        return True, {}, video_path_str, evaluation_rubric, input_text

    async def evaluate_manim_video(self, state: ManimAgentState) -> Dict[str, Any]:
        """Evaluates the Manim video using frame extraction and a multimodal LLM."""
        run_output_dir = Path(state["run_output_dir"])
        current_attempt = state.get("attempt_number", 0) + 1

        log_run_details(
            run_output_dir,
            current_attempt,
            self.NODE_NAME,
            "Node Entry",
            f"Starting {self.NODE_NAME}...",
        )

        checks_passed, failure_updates, video_path, evaluation_rubric, input_text = (
            await self._perform_pre_checks(state, run_output_dir, current_attempt)
        )
        if not checks_passed:
            return failure_updates

        updates_to_state: Dict[str, Any] = {
            "evaluation_result": None,
            "evaluation_passed": None,
        }

        log_run_details(
            run_output_dir,
            current_attempt,
            self.NODE_NAME,
            "Prepare Payload",
            f"Preparing frames from video: {video_path}",
        )
        frame_payloads = await self._prepare_video_payload(video_path)

        if frame_payloads is None:
            error_message = f"Evaluation failed: Could not prepare frames from video {video_path} (check size/ffmpeg errors)."
            log_run_details(
                run_output_dir,
                current_attempt,
                self.NODE_NAME,
                "Payload Error",
                error_message,
                is_error=True,
            )
            updates_to_state["evaluation_result"] = {"feedback": error_message, "passed": False}
            updates_to_state["evaluation_passed"] = False
            return updates_to_state

        if not frame_payloads:
            warning_message = f"Evaluation Warning: No frames extracted/selected from video {video_path}. Evaluation might be inaccurate."
            log_run_details(
                run_output_dir,
                current_attempt,
                self.NODE_NAME,
                "Payload Warning",
                warning_message,
                is_error=True,  # Log as error because evaluation can't proceed meaningfully
            )
            updates_to_state["evaluation_result"] = {"feedback": warning_message, "passed": False}
            updates_to_state["evaluation_passed"] = False
            return updates_to_state

        prompt_text = self.prompt_template.format(
            evaluation_rubric=evaluation_rubric, input_text=input_text or "[Input text missing]"
        )
        message_content = [prompt_text] + frame_payloads
        message = HumanMessage(content=message_content)

        log_run_details(
            run_output_dir,
            current_attempt,
            self.NODE_NAME,
            "LLM Prompt",
            prompt_text,
        )
        try:
            logger.info(f"Calling Evaluation LLM: {self.llm_eval_client.__class__.__name__}")
            response = await self.llm_eval_client.ainvoke([message])
            response_text = response.content if hasattr(response, "content") else str(response)
            log_run_details(
                run_output_dir,
                current_attempt,
                self.NODE_NAME,
                "LLM Response",
                response_text,
            )

            # Parse the response using the robust JsonOutputParser
            try:
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    self.NODE_NAME,
                    "Parse Result",
                    "Attempting to parse JSON output using JsonOutputParser.",
                )
                parsed_result = self.parser.parse(response_text)

                # Check the type of the parsed result
                if isinstance(parsed_result, EvaluationResult):
                    logger.info(
                        f"Evaluation result (Pydantic): Pass={parsed_result.passed}, Feedback: {parsed_result.feedback[:100]}..."
                    )
                    updates_to_state["evaluation_result"] = (
                        parsed_result.model_dump()
                    )  # Store as dict
                    updates_to_state["evaluation_passed"] = parsed_result.passed
                elif isinstance(parsed_result, dict):
                    logger.warning(
                        f"Parsed result as dict (Pydantic validation might have partially failed): {parsed_result}"
                    )
                    # Attempt to access keys, default to False/Error message if missing
                    passed = parsed_result.get("passed", False)
                    feedback = parsed_result.get("feedback", "Error: Keys missing in parsed dict")
                    updates_to_state["evaluation_result"] = parsed_result  # Store the dict directly
                    updates_to_state["evaluation_passed"] = passed
                    # Log potentially missing keys
                    if "passed" not in parsed_result or "feedback" not in parsed_result:
                        logger.error(
                            f"Parsed dict missing 'passed' or 'feedback' key: {parsed_result}"
                        )
                        log_run_details(
                            run_output_dir,
                            current_attempt,
                            self.NODE_NAME,
                            "Parse Warning",
                            f"Parsed dict missing 'passed' or 'feedback' key: {parsed_result}",
                            is_error=True,
                        )
                        # Override feedback if keys missing
                        updates_to_state["evaluation_result"][
                            "feedback"
                        ] = f"Error: Parsed dict missing keys. Original dict: {parsed_result}"
                        updates_to_state["evaluation_passed"] = (
                            False  # Ensure failure if keys missing
                        )

                    logger.info(
                        f"Evaluation result (dict): Pass={passed}, Feedback: {str(feedback)[:100]}..."
                    )
                else:
                    # This case should ideally not happen if parser works as expected
                    raise TypeError(f"Unexpected parsed result type: {type(parsed_result)}")

                log_run_details(
                    run_output_dir,
                    current_attempt,
                    self.NODE_NAME,
                    "Node Completion",
                    f"Evaluation completed. Pass: {updates_to_state['evaluation_passed']}",
                )
            except OutputParserException as pe:
                error_message = f"Evaluation failed: Could not parse LLM response using JsonOutputParser: {pe}\nRaw Response: {response_text[:500]}..."
                logger.error(error_message)
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    self.NODE_NAME,
                    "Parse Error",
                    error_message,
                    is_error=True,
                )
                updates_to_state["evaluation_result"] = {"feedback": error_message, "passed": False}
                updates_to_state["evaluation_passed"] = False

        except Exception as e:
            error_message = f"Error during evaluation LLM call: {e}"
            logger.error(error_message, exc_info=True)
            log_run_details(
                run_output_dir,
                current_attempt,
                self.NODE_NAME,
                "LLM Error",
                f"{error_message}\n{traceback.format_exc()}",
                is_error=True,
            )
            updates_to_state["evaluation_result"] = {"feedback": error_message, "passed": False}
            updates_to_state["evaluation_passed"] = False

        return updates_to_state

    # MODIFIED: Make async
    async def _prepare_video_payload(self, video_path_str: str) -> Optional[List[Dict[str, Any]]]:
        """Prepares the video payload: checks size, extracts frames, selects frames, encodes."""
        video_path = Path(video_path_str)
        if not video_path.is_file():
            logger.error(f"Video file not found: {video_path}")
            return None

        # Check video size
        video_size = video_path.stat().st_size
        if video_size > MAX_VIDEO_SIZE_BYTES:
            logger.warning(
                f"Video size ({video_size / 1024**2:.2f} MB) exceeds limit ({VIDEO_EVAL_MAX_SIZE_MB} MB). Skipping frame extraction."
            )
            # Return a specific indicator? Or just None? Let's return None to signal failure.
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Extract frames asynchronously
            # MODIFIED: Directly await the async function
            extracted_frames = await self._extract_frames(
                str(video_path), temp_path, VIDEO_EVAL_FRAMES_PER_SECOND
            )

            if extracted_frames is None:
                logger.error("Frame extraction failed.")
                return None
            if not extracted_frames:
                logger.warning("No frames were extracted.")
                return []

            # Select frames (e.g., first N frames)
            # Ensure we don't exceed the max frames *after* extraction
            selected_frames = extracted_frames
            if len(selected_frames) > VIDEO_EVAL_MAX_FRAMES:
                logger.warning(
                    f"Extracted {len(extracted_frames)} frames, exceeding limit {VIDEO_EVAL_MAX_FRAMES}. Selecting first {VIDEO_EVAL_MAX_FRAMES}."
                )
                selected_frames = extracted_frames[:VIDEO_EVAL_MAX_FRAMES]
            else:
                logger.info(
                    f"Selected {len(selected_frames)} frames out of {len(extracted_frames)} (max: {VIDEO_EVAL_MAX_FRAMES})"
                )

            # Encode selected frames
            payloads = []
            for frame_path in selected_frames:
                mime_type, _ = mimetypes.guess_type(frame_path)
                if not mime_type or not mime_type.startswith("image/"):
                    logger.warning(f"Could not determine image mime type for {frame_path}")
                    continue
                base64_data = self._encode_image_to_base64(frame_path)
                if base64_data:
                    payloads.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                        }
                    )
            return payloads

    def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Encodes an image file to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}", exc_info=True)
            return None
