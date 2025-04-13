import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from config import base_config as base_cfg

# Load environment variables early, consider moving this to main app entry point later
dotenv.load_dotenv()


def create_llm_client(provider: str, model_name: str, temperature: float) -> BaseChatModel:
    """
    Factory function to create an LLM client based on the specified provider.

    Args:
        provider: The name of the LLM provider (e.g., 'google').
        model_name: The specific model name to use.
        temperature: The sampling temperature for the model.

    Returns:
        An instance of a LangChain chat model client.

    Raises:
        ValueError: If the provider is unsupported or the required API key is missing.
    """
    if provider.lower() == "google":
        api_key = base_cfg.get_gemini_api_key()
        if not api_key:
            raise ValueError(
                "Google provider selected, but Gemini API key not found. "
                "Set GOOGLE_API_KEY environment variable."
            )
        client = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, temperature=temperature
        )
        print(
            f"Initialized Google LLM Client: model={client.model}, "
            f"temperature={client.temperature}"
        )
        return client
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
