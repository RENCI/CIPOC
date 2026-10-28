"""Print the first Chat Completions request the orchestrator sends, and its response.

The request is the JSON body ``ChatOpenAI`` builds, i.e. what
``FoundryModelsTransport`` hands ``to_request``; the response is the endpoint's
reply, i.e. the shape ``to_response`` must produce. The orchestrator runs
against the configured endpoint with ``use_responses_api`` forced off. Only the
first request is forwarded; later ones are refused, so the run stops after one
model call and its failure is expected.

    PYTHONPATH=src python -m scripts.print_first_exchange \
        [--notes tests/fixtures/note_bundle.json] [--config config/config.yaml] [--max-chars 300]

Headers are not printed (they carry the API key). Use synthetic notes: the
request contains the note text.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import httpx

from cipoc.agents import OrchestratorAgent
from cipoc.models import OrchestratorRunError
from cipoc.utils import CipocConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTES = REPO_ROOT / "tests" / "fixtures" / "note_bundle.json"


class _FirstExchangeTransport(httpx.BaseTransport):
    """Forwards the first request to the network and refuses the rest."""

    def __init__(self) -> None:
        self._network = httpx.HTTPTransport()
        self._lock = threading.Lock()
        self.request: httpx.Request | None = None
        self.response: httpx.Response | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            first = self.request is None
            if first:
                self.request = request
        if not first:
            # 400 is not retried, so the run stops quickly.
            return httpx.Response(400, json={"error": {"message": "Probe forwards only the first request."}})
        response = self._network.handle_request(request)
        response.read()
        self.response = response
        return response

    def close(self) -> None:
        self._network.close()


def _shorten(value: Any, limit: int) -> Any:
    if isinstance(value, str) and limit and len(value) > limit:
        return f"{value[:limit]}... [{len(value) - limit} more chars]"
    if isinstance(value, dict):
        return {key: _shorten(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_shorten(item, limit) for item in value]
    return value


def _print_body(content: bytes, limit: int) -> None:
    try:
        body = json.loads(content)
    except ValueError:
        print(content.decode("utf-8", errors="replace"))
        return
    print(json.dumps(_shorten(body, limit), indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES, help="JSON array of clinical notes.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML (default: config/config.yaml).")
    parser.add_argument(
        "--max-chars", type=int, default=300,
        help="Truncate printed strings to this many characters; 0 prints them in full.",
    )
    args = parser.parse_args()

    probe = _FirstExchangeTransport()
    with httpx.Client(transport=probe) as client:
        config = CipocConfig.load(args.config, model_kwargs={"http_client": client})
        # The transport serves only Chat Completions. No SDK retries, so the
        # refused requests fail on the first attempt.
        config.defaults["use_responses_api"] = False
        config.defaults["max_retries"] = 0
        for override in config.agents.values():
            if override:
                override.pop("use_responses_api", None)
                override.pop("max_retries", None)

        notes = json.loads(args.notes.read_text(encoding="utf-8"))
        try:
            OrchestratorAgent(config=config).run(notes, progress=False)
        except OrchestratorRunError:
            pass  # expected: every request after the first is refused

    if probe.request is None:
        print("The orchestrator sent no model request.")
        return 1
    print(f"=== REQUEST: {probe.request.method} {probe.request.url.path} ===")
    _print_body(probe.request.content, args.max_chars)
    print()
    if probe.response is None:
        print("=== RESPONSE: none (the request failed before a response arrived) ===")
        return 1
    print(f"=== RESPONSE: HTTP {probe.response.status_code} ===")
    _print_body(probe.response.content, args.max_chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
