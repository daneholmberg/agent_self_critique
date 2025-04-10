from langgraph.graph import END, StateGraph
from typing import Callable
from langgraph.pregel import Pregel  # Assuming CompiledGraph is Pregel

from core.graph_state import GraphState


def decide_after_validation(state: GraphState) -> str:
    """
    Conditional edge determining the next step after code validation.

    Args:
        state: The current graph state.

    Returns:
        The name of the next node ('generate_output' or 'evaluate_output') or END
        if the maximum iteration limit is reached due to validation errors.
    """
    if state["validation_error"]:
        if state["iteration"] >= state["max_iterations"]:
            print("Maximum validation iterations reached. Ending graph.")
            return END
        else:
            print("Validation failed. Returning to generate output.")
            return "generate_output"
    else:
        print("Validation successful. Proceeding to evaluate output.")
        return "evaluate_output"


def decide_after_evaluation(state: GraphState) -> str:
    """
    Conditional edge determining the next step after code evaluation.

    Args:
        state: The current graph state.

    Returns:
        The name of the next node ('generate_output') or END if the evaluation
        was successful or the maximum iteration limit is reached.
    """
    if state.get("evaluation_passed", False):  # Default to False if key doesn't exist
        print("Evaluation successful.")
        return END
    else:
        if state["iteration"] >= state["max_iterations"]:
            print("Maximum evaluation iterations reached. Ending graph.")
            return END
        else:
            print("Evaluation failed. Returning to generate output.")
            return "generate_output"


def build_graph(
    generate_node_func: Callable[[GraphState], dict],
    validate_node_func: Callable[[GraphState], dict],
    evaluate_node_func: Callable[[GraphState], dict],
) -> Pregel:
    """
    Builds and compiles the generic LangGraph workflow.

    Args:
        generate_node_func: The function to execute for the 'generate_output' node.
        validate_node_func: The function to execute for the 'validate_output' node.
        evaluate_node_func: The function to execute for the 'evaluate_output' node.

    Returns:
        The compiled LangGraph application.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("generate_output", generate_node_func)
    workflow.add_node("validate_output", validate_node_func)
    workflow.add_node("evaluate_output", evaluate_node_func)

    # Set entry point
    workflow.set_entry_point("generate_output")

    # Add edges
    workflow.add_edge("generate_output", "validate_output")

    # Add conditional edges
    workflow.add_conditional_edges(
        "validate_output",
        decide_after_validation,
        {
            "evaluate_output": "evaluate_output",
            "generate_output": "generate_output",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "evaluate_output", decide_after_evaluation, {"generate_output": "generate_output", END: END}
    )

    # Compile the graph
    app = workflow.compile()
    return app
