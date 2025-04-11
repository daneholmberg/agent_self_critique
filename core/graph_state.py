from typing import List, Optional, TypedDict


class GraphState(TypedDict):
    """Generic state for LangGraph agents.

    Represents the state of our graph.

    Attributes:
        input_text: The primary text input (e.g., script segment, draft).
        input_metadata: Optional additional context (timestamps, keywords).
        context_doc: Content of the context/style guide document.
        rubric: Content of the evaluation rubric.
        max_iterations: Maximum number of generation/validation attempts.
        iteration: Current iteration number (1-based).
        general_context: Optional general context/instructions for the generator.
        previous_code_attempt: Optional previous code provided for enhancement.
        enhancement_request: Optional description of the requested enhancements.
        final_command: Optional final instruction/prompt for the generator.
        generated_output: Agent's output in the current iteration (code, text).
        validation_error: Error message from validation/execution (e.g., stderr).
        validated_artifact_path: Path to a validated artifact (e.g., video file), if any.
        evaluation_feedback: Feedback from the AI evaluation step.
        evaluation_passed: Boolean indicating if the evaluation passed.
        error_history: Accumulated validation/execution errors across iterations.
        evaluation_history: Accumulated evaluation feedback across iterations.
        final_output_path: Path where the final primary output (code/text) is saved.
        final_artifact_path: Path where the final artifact (e.g., video) is saved.
        infrastructure_error: An error that occurred in the agent's supporting code, not the generated output.

        # New fields for run-specific outputs and debug flags
        run_output_dir: str # Absolute path to the base directory for *this* run's outputs
        save_generated_code: bool # Flag indicating whether to save generated code permanently
    """

    input_text: str
    input_metadata: Optional[str]
    context_doc: str
    rubric: str
    max_iterations: int
    iteration: int
    general_context: Optional[str]
    previous_code_attempt: Optional[str]
    enhancement_request: Optional[str]
    final_command: Optional[str]
    generated_output: Optional[str]
    validation_error: Optional[str]
    validated_artifact_path: Optional[str]
    evaluation_feedback: Optional[str]
    evaluation_passed: Optional[bool]
    error_history: List[str]
    evaluation_history: List[str]
    final_output_path: Optional[str]
    final_artifact_path: Optional[str]
    infrastructure_error: Optional[str]

    # New fields
    run_output_dir: str
    save_generated_code: bool
