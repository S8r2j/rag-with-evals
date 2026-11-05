import openai

from app.core.config import Settings
from app.llm.base import ChatError, ChatModel, GeneratedAnswer, Source, TransientChatError
from app.llm.extractive import ExtractiveChatModel
from app.llm.openai import OpenAIChatModel

__all__ = [
    "ChatError",
    "ChatModel",
    "ExtractiveChatModel",
    "GeneratedAnswer",
    "OpenAIChatModel",
    "Source",
    "TransientChatError",
    "build_chat_model",
]


def build_chat_model(settings: Settings) -> ChatModel:
    if settings.chat_provider == "extractive":
        return ExtractiveChatModel()
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is required when CHAT_PROVIDER is 'openai'")
    client = openai.AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_seconds,
        max_retries=2,
    )
    return OpenAIChatModel(
        client,
        model=settings.chat_model,
        max_output_tokens=settings.chat_max_output_tokens,
        reasoning_effort=settings.chat_reasoning_effort,
    )
