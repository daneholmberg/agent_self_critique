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
        generated_output: Agent's output in the current iteration (code, text).
        validation_error: Error message from validation/execution (e.g., stderr).
        validated_artifact_path: Path to a validated artifact (e.g., video file), if any.
        evaluation_feedback: Feedback from the AI evaluation step.
        evaluation_passed: Boolean indicating if the evaluation passed.
        error_history: Accumulated validation/execution errors across iterations.
        evaluation_history: Accumulated evaluation feedback across iterations.
        final_output_path: Path where the final primary output (code/text) is saved.
        final_artifact_path: Path where the final artifact (e.g., video) is saved.
    """

    input_text: str
    input_metadata: Optional[str]
    context_doc: str
    rubric: str
    max_iterations: int
    iteration: int
    generated_output: Optional[str]
    validation_error: Optional[str]
    validated_artifact_path: Optional[str]
    evaluation_feedback: Optional[str]
    evaluation_passed: Optional[bool]
    error_history: List[str]
    evaluation_history: List[str]
    final_output_path: Optional[str]
    final_artifact_path: Optional[str]
