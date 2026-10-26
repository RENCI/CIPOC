"""In-process Chat Completions transport for Palantir Foundry's models API.

Foundry Python Transforms receive language models as injected objects
(``OpenAiGptChatLanguageModelInput``) rather than as an HTTP endpoint. An
``httpx.Client`` built on :class:`FoundryModelsTransport` lets the unmodified
``ChatOpenAI`` stack run against those objects: the OpenAI SDK builds the Chat
Completions request, the transport passes its JSON body through caller-supplied
translators around ``model.create_chat_completion``, and returns the translated
completion as the HTTP response. Nothing opens a socket, so ``base_url`` only
has to pass validation; an ``.invalid`` host fails loudly if a request ever
bypasses the transport.

Nothing here imports Foundry. Models and translators are duck-typed, so the
module stays importable and testable anywhere.

The OpenAI SDK wraps any exception escaping a transport in a retried
``APIConnectionError``, so every failure is returned as an HTTP status chosen
for :func:`~cipoc.llm.retry_on_transient`:

- unsupported paths (including the Responses API), streaming, unrouted model
  names, and request translation failures return 4xx and fail on the first
  attempt;
- a failed model call returns the HTTP status its exception carries
  (``status_code`` or ``response.status_code``), otherwise 502, which retries;
- response translation failures return 400: a translator bug must not cost
  ``max_attempts`` model calls.

Raise :class:`ChatCompletionStatusError` from either translator or the model
call to choose the status explicitly, e.g. 429 for a throttling exception that
carries no status.

httpx timeouts govern network I/O only and do not bound
``create_chat_completion``; the models API's own timeout applies. The transport
holds no per-request state, but concurrent graph branches call the model
objects from several threads at once.

Usage inside a transform::

    client = httpx.Client(transport=FoundryModelsTransport(
        model, to_request=to_foundry_request, to_response=to_openai_response,
    ))
    config = load_config("config/config.yaml", model_kwargs={"http_client": client})
    result = OrchestratorAgent(config=config).run(notes, progress=False)
"""

from collections.abc import Callable, Mapping
import json
from typing import Any, Protocol

import httpx


_CHAT_COMPLETIONS_PATH = "/chat/completions"


class ChatCompletionModel(Protocol):
    """The part of a Foundry language model the transport calls."""

    def create_chat_completion(self, request: Any) -> Any:
        ...


class ChatCompletionStatusError(Exception):
    """Raised by a translator or model call to choose the HTTP status returned."""

    def __init__(self, status_code: int, message: str):
        if type(status_code) is not int or not 400 <= status_code <= 599:
            raise ValueError("status_code must be an HTTP error status (400-599).")
        super().__init__(message)
        self.status_code = status_code


def _error_response(status_code: int, kind: str, message: str) -> httpx.Response:
    # OpenAI's error envelope, so the SDK surfaces ``message`` on the raised error.
    return httpx.Response(status_code, json={"error": {"message": message, "type": kind, "code": None}})


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _carried_status(error: BaseException) -> int | None:
    for source in (error, getattr(error, "response", None)):
        status = getattr(source, "status_code", None)
        if type(status) is int and 400 <= status <= 599:
            return status
    return None


class FoundryModelsTransport(httpx.BaseTransport):
    """Serve OpenAI Chat Completions requests from Foundry models API objects.

    ``models`` is one model, used for every request, or a mapping from the
    request's ``model`` name to a model, for agents configured with different
    models. ``to_request`` turns a Chat Completions request body into the
    model's request object; ``to_response`` turns the model's response into a
    Chat Completions response body (a mapping or a Pydantic model), whose
    ``model`` defaults to the requested name. Responses should carry
    ``finish_reason`` and ``usage`` so structured parsing and token
    observability keep working.
    """

    def __init__(
        self,
        models: ChatCompletionModel | Mapping[str, ChatCompletionModel],
        *,
        to_request: Callable[[dict[str, Any]], Any],
        to_response: Callable[[Any], Any],
    ) -> None:
        candidates = list(models.values()) if isinstance(models, Mapping) else [models]
        if not candidates:
            raise ValueError("At least one model is required.")
        if not all(callable(getattr(model, "create_chat_completion", None)) for model in candidates):
            raise TypeError("Models must provide create_chat_completion(request).")
        self._models = models
        self._to_request = to_request
        self._to_response = to_response

    def _model_for(self, name: Any) -> ChatCompletionModel | None:
        if isinstance(self._models, Mapping):
            return self._models.get(name) if isinstance(name, str) else None
        return self._models

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.rstrip("/").endswith(_CHAT_COMPLETIONS_PATH):
            return _error_response(
                404, "unsupported_endpoint",
                f"Only POST {_CHAT_COMPLETIONS_PATH} is served; set use_responses_api: false.",
            )
        try:
            body = json.loads(request.read())
        except ValueError as error:
            return _error_response(400, "invalid_request_error", f"Request body is not JSON: {error}")
        if not isinstance(body, dict):
            return _error_response(400, "invalid_request_error", "Request body must be a JSON object.")
        if body.get("stream"):
            return _error_response(400, "invalid_request_error", "Streaming is not supported by the models API transport.")
        model = self._model_for(body.get("model"))
        if model is None:
            return _error_response(404, "model_not_found", f"No models API model is routed for {body.get('model')!r}.")

        try:
            upstream_request = self._to_request(body)
        except ChatCompletionStatusError as error:
            return _error_response(error.status_code, "request_translation_error", str(error))
        except Exception as error:
            return _error_response(400, "request_translation_error", f"Could not translate the request: {_describe(error)}")

        try:
            upstream_response = model.create_chat_completion(upstream_request)
        except Exception as error:
            return _error_response(
                _carried_status(error) or 502, "upstream_error", f"Models API call failed: {_describe(error)}",
            )

        try:
            completion = self._to_response(upstream_response)
            if hasattr(completion, "model_dump"):
                completion = completion.model_dump(mode="json")
            completion = dict(completion)
            if not completion.get("model"):
                completion["model"] = body.get("model")
            return httpx.Response(200, json=completion)
        except ChatCompletionStatusError as error:
            return _error_response(error.status_code, "response_translation_error", str(error))
        except Exception as error:
            return _error_response(400, "response_translation_error", f"Could not translate the response: {_describe(error)}")


__all__ = [
    "ChatCompletionModel",
    "ChatCompletionStatusError",
    "FoundryModelsTransport",
]
