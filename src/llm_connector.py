from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from threading import Lock
# from dataclasses import dataclass
# from collections.abc import Sequence
from pydantic import BaseModel, Field, field_validator
from typing import Any


class LLMError(RuntimeError):
    """All configured API keys failed."""


# @dataclass(frozen=True)
class ProviderConfig(BaseModel):
    name: str = "test"
    base_url: str = Field(min_length=3)
    model: str = Field(min_length=1)
    api_keys: tuple[str, ...] = Field(repr=False)

    @field_validator("api_keys", mode="before")
    @classmethod
    def fix_api_keys(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        else:
            return tuple(value)


class LLMConnector:
    """
    Connector with providers with a list of keys.
    """

    def __init__(self, config: ProviderConfig) -> None:

        if not config.api_keys:
            raise ValueError(
                f"No API keys configured for {config.name}"
            )

        self.config = config
        self._clients = [
            OpenAI(
                api_key=key,
                base_url=config.base_url,
                max_retries=0,
                timeout=60.0,
            )
            for key in config.api_keys
        ]

        self._index = 0
        self._lock = Lock()

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True

        if isinstance(exc, APIStatusError):
            return exc.status_code in {
                401, 403, 408, 429, 500, 502, 503, 504
            }

        return False

    def _next_start_index(self) -> int:
        """Round-robin starting point for requests."""
        with self._lock:
            index = self._index
            self._index = (self._index + 1) % len(self._clients)
            return index

    def chat_raw(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        start = self._next_start_index()
        last_error: Exception | None = None

        for offset in range(len(self._clients)):
            client = self._clients[
                (start + offset) % len(self._clients)
            ]

            try:
                return client.chat.completions.with_raw_response.create(
                    model=model or self.config.model,
                    messages=messages,
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc

                if not self._should_retry(exc):
                    raise

        raise LLMError(
            f"All {len(self._clients)} API keys failed "
            f"for provider {self.config.name}"
        ) from last_error

    def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        start = self._next_start_index()
        last_error: Exception | None = None

        for offset in range(len(self._clients)):
            client = self._clients[
                (start + offset) % len(self._clients)
            ]

            try:
                return client.chat.completions.create(
                    model=model or self.config.model,
                    messages=messages,
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc

                if not self._should_retry(exc):
                    raise

        raise LLMError(
            f"All {len(self._clients)} API keys failed "
            f"for provider {self.config.name}"
        ) from last_error


# class LLMConnector:
#     def __init__(self, api_keys_config: dict[str, list[str]]):
#         """
#         Initialize the connector with a dictionary mapping providers
#         to a list of keys.
#         Example:
#             api_keys_config = {
#                 "groq": ["gsk_key1...", "gsk_key2..."],
#                 "gemini": ["AIzaSyKey1...", "AIzaSyKey2..."],
#                 "openrouter": ["sk-or-v1..."]
#             }
#         """
#         self.api_keys = api_keys_config
