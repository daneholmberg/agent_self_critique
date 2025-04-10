import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config import base_config as base_cfg

# Load environment variables (especially for API keys if not set globally)
# Done centrally here to avoid scattering dotenv.load_dotenv() calls
dotenv.load_dotenv()


def get_llm_clients() -> tuple[ChatGoogleGenerativeAI, ChatGoogleGenerativeAI]:
    """
    Initializes and returns the Gemini clients for text generation and evaluation.

    Ensures the API key is loaded and configured for both clients.

    Returns:
        tuple[ChatGoogleGenerativeAI, ChatGoogleGenerativeAI]: A tuple containing
            the text generation client and the evaluation client.

    Raises:
        ValueError: If the Gemini API key is not found in the environment.
    """
    api_key = base_cfg.get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY environment variable.")

    # Client for code generation (potentially higher temperature)
    llm_text_client = ChatGoogleGenerativeAI(
        model=base_cfg.GEMINI_DEFAULT_MODEL_NAME,
        google_api_key=api_key,
        temperature=0.7,
    )
    print(f"Initialized Text LLM: {llm_text_client.model}")

    # Client for evaluation (ensure model supports video, lower temperature)
    # Ensure the evaluation model explicitly supports vision/video if different from default
    eval_model_name = (
        base_cfg.GEMINI_DEFAULT_MODEL_NAME
    )  # Or potentially a specific vision model from config
    llm_eval_client = ChatGoogleGenerativeAI(
        model=eval_model_name,
        google_api_key=api_key,
        temperature=0.2,  # Lower temperature for deterministic evaluation
    )
    print(f"Initialized Evaluation LLM: {llm_eval_client.model}")

    return llm_text_client, llm_eval_client


# Example of how you might use this (in your graph setup):
# from agents.manim_agent.llm_clients import get_llm_clients
# text_client, eval_client = get_llm_clients()
