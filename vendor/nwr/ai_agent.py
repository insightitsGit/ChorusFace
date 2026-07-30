"""Command-line AI client for the Neural World Runtime bridge.

Translates a natural-language instruction into world commands using any
OpenAI-compatible chat completions endpoint, then submits the result to the
running simulation. Also supports sending hand-written command files and
inspecting the world without a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

from ai_commands import (
    COMMAND_SCHEMA,
    DEFAULT_PRIORITY,
    SYSTEM_PROMPT,
    CommandError,
    compile_request,
)
from bds_format import PRIORITY_LEVELS, PRIORITY_NAMES

DEFAULT_SERVER: Final = "http://127.0.0.1:8765"
DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
DEFAULT_MODEL: Final = "gpt-4o-mini"
REQUEST_TIMEOUT: Final = 60.0


class AgentError(RuntimeError):
    """A user-facing failure in the client workflow."""


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        return _run(arguments)
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive the video playground with an AI model or a command file.",
    )
    parser.add_argument(
        "instruction",
        nargs="?",
        help="Natural-language request, for example 'flood the lower third'",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("NWR_AI_SERVER_URL", DEFAULT_SERVER),
        help=f"Bridge base URL (default {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("NWR_AI_TOKEN", ""),
        help="Bridge access token printed by main.py",
    )
    parser.add_argument(
        "--commands-file",
        type=Path,
        help="Send a JSON command file instead of calling a model",
    )
    parser.add_argument(
        "--observe",
        action="store_true",
        help="Include the current world observation in the model prompt",
    )
    parser.add_argument(
        "--state",
        action="store_true",
        help="Print the world observation and exit",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save a PNG of the latest frame and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without submitting them",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NWR_LLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NWR_LLM_MODEL", DEFAULT_MODEL),
        help=f"Model name (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NWR_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="API key for the model endpoint",
    )
    return parser


def _run(arguments: argparse.Namespace) -> int:
    server = str(arguments.server).rstrip("/")
    token = str(arguments.token)

    if arguments.screenshot is not None:
        image = _request(server, "/screenshot", token=token, raw=True)
        arguments.screenshot.write_bytes(image)
        print(f"Saved {arguments.screenshot}")
        return 0

    if arguments.state:
        print(json.dumps(_request(server, "/state", token=token), indent=2))
        return 0

    # The bridge narrows the grammar to the authority it grants this caller, so
    # ask it rather than assuming: prompting a model with commands it is not
    # allowed to send, or approving them locally, only moves the refusal later.
    schema: dict[str, Any] | None = None
    if arguments.commands_file is not None:
        payload = _load_json_file(arguments.commands_file)
        if not arguments.dry_run:
            schema = _request(server, "/schema", token=token)
    elif arguments.instruction:
        schema = _request(server, "/schema", token=token)
        observation = (
            _request(server, "/state", token=token) if arguments.observe else None
        )
        payload = _generate_commands(
            arguments.instruction,
            observation=observation,
            schema=schema,
            base_url=str(arguments.base_url).rstrip("/"),
            model=str(arguments.model),
            api_key=str(arguments.api_key),
        )
    else:
        raise AgentError(
            "Provide an instruction, --commands-file, --state, or --screenshot"
        )

    authority = _authority(schema)
    try:
        operations = compile_request(payload, default_priority=authority)
    except CommandError as exc:
        raise AgentError(f"Generated commands are invalid: {exc}") from None

    print(json.dumps(payload, indent=2))
    print(
        f"Compiles to {len(operations)} GPU operations "
        f"at '{PRIORITY_NAMES[authority]}' authority"
        + ("" if schema is not None else " (local default; bridge not consulted)")
    )
    if arguments.dry_run:
        return 0

    result = _request(server, "/commands", token=token, body=payload)
    print(json.dumps(result, indent=2))
    return 0


def _authority(schema: dict[str, Any] | None) -> int:
    """The authority the bridge reported, or the local default without one."""
    name = (schema or {}).get("caller_authority")
    if isinstance(name, str) and name in PRIORITY_LEVELS:
        return PRIORITY_LEVELS[name]
    return DEFAULT_PRIORITY


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AgentError(f"Could not read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise AgentError(f"{path} is not valid JSON: {exc}") from None


def _generate_commands(
    instruction: str,
    *,
    observation: dict[str, Any] | None,
    schema: dict[str, Any] | None = None,
    base_url: str,
    model: str,
    api_key: str,
) -> Any:
    if not api_key:
        raise AgentError(
            "No model API key. Set NWR_LLM_API_KEY or OPENAI_API_KEY, or use "
            "--commands-file to send commands directly."
        )

    grammar = schema if schema is not None else COMMAND_SCHEMA
    user_content = [f"Request: {instruction}", "", "Command grammar:", json.dumps(grammar)]
    if observation is not None:
        user_content.extend(["", "Current world observation:", json.dumps(observation)])

    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_content)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    response = _http_json(
        f"{base_url}/chat/completions",
        body=request_body,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentError(f"Unexpected model response shape: {exc}") from None
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise AgentError(f"Model did not return JSON: {exc}\n{content}") from None


def _request(
    server: str,
    path: str,
    *,
    token: str,
    body: Any | None = None,
    raw: bool = False,
) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{server}{path}"
    if raw:
        return _http_bytes(url, headers=headers)
    return _http_json(url, body=body, headers=headers)


def _http_json(
    url: str,
    *,
    body: Any | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    payload = _http_bytes(url, body=body, headers=headers)
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentError(f"{url} returned a non-JSON response: {exc}") from None


def _http_bytes(
    url: str,
    *,
    body: Any | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    data = None
    request_headers = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise AgentError(f"{url} failed with HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise AgentError(
            f"Could not reach {url}: {exc.reason}. Start the playground with "
            "--ai-server."
        ) from None


if __name__ == "__main__":
    raise SystemExit(main())
