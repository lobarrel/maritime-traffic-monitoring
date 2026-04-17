import json
import os
import secrets
from typing import Any, Sequence

import httpx
from ollama import Client
from ollama._client import _copy_tools
from ollama._types import ChatResponse, ResponseError

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
MODEL_NAME = os.environ.get("VLM_MODEL", "gemini-3-flash-preview:cloud")

STAC_API_URL = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-l2a"

SEARCH_RADIUS_KM = 10
MAX_IMAGES = 4
MAX_CLOUD_COVER = 30  # percent

INVESTIGATOR_MAX_ITERATIONS = 10
# How many distinct Sentinel-2 passes (calendar dates) explore_direction fetches
# for the same area (newest-first).  Enables before/after comparison at a suspected
# cause.  Set 1 to disable multi-temporal explore.  Hard cap: EXPLORE_TEMPORAL_MAX.
EXPLORE_MAX_TEMPORAL_IMAGES = int(os.environ.get("EXPLORE_MAX_TEMPORAL_IMAGES", str(MAX_IMAGES)))
# Max distinct dates per explore (defaults to MAX_IMAGES; raise independently of monitor).
EXPLORE_TEMPORAL_MAX = int(os.environ.get("EXPLORE_TEMPORAL_MAX", str(MAX_IMAGES)))
# Longest side of JPEGs sent to the VLM (after cropping the AOI).  Higher =
# more detail and larger API payloads.  If Ollama Cloud returns “request body
# too large”, lower this or ``VLM_JPEG_QUALITY``.  Local Ollama often allows 4096+.
VLM_MAX_IMAGE_PX = int(os.environ.get("VLM_MAX_IMAGE_PX", "2048"))
VLM_JPEG_QUALITY = int(os.environ.get("VLM_JPEG_QUALITY", "90"))

OUTPUT_DIR = os.environ.get("VLM_OUTPUT_DIR", "./output")
TEMP_DIR = os.environ.get("VLM_TEMP_DIR", "./tmp_images")

# Gemini 3 / some Ollama Cloud models require ``thought_signature`` on each
# assistant tool_call when continuing the conversation. The ollama Python
# ``Message`` model does not define this field, so it can be dropped from
# history and the next /api/chat request returns 400.  Preserve real values
# when present; otherwise inject a placeholder (see Ollama issue #14567).
_INJECT_TOOL_THOUGHT_SIGNATURE = os.environ.get(
    "OLLAMA_INJECT_TOOL_THOUGHT_SIGNATURE", "1"
).strip().lower() not in ("0", "false", "no")
_TOOL_THOUGHT_SIGNATURE_PLACEHOLDER = os.environ.get(
    "OLLAMA_TOOL_THOUGHT_SIGNATURE_PLACEHOLDER",
    "skip_thought_signature_validator",
).strip() or "skip_thought_signature_validator"

_client: Client | None = None


def investigator_uses_native_tools(model: str) -> bool:
    """Use Ollama ``/api/chat`` with a ``tools`` field (native function calling).

    Gemini 3 on Ollama Cloud omits ``thought_signature`` on tool calls in API
    responses, so multi-turn tool use returns 400 (ollama#14567, #15109).  For
    those models we use JSON tool lines in plain text instead.

    * ``INVESTIGATOR_NATIVE_TOOLS`` — ``auto`` (default), ``always``, ``never``.
    """
    mode = os.environ.get("INVESTIGATOR_NATIVE_TOOLS", "auto").strip().lower()
    if mode in ("1", "true", "yes", "always"):
        return True
    if mode in ("0", "false", "no", "never"):
        return False
    return "gemini-3" not in model.lower()


def _ollama_message_to_dict(msg: Any) -> dict[str, Any]:
    if isinstance(msg, dict):
        return dict(msg)
    return msg.model_dump(exclude_none=True)


def _fix_tool_call_arguments_objects(msg_dict: dict[str, Any]) -> None:
    """Gemini prefers tool arguments as objects, not JSON strings (Ollama #14567)."""
    if msg_dict.get("role") != "assistant" or not msg_dict.get("tool_calls"):
        return
    for tc in msg_dict["tool_calls"]:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
            raw = fn["arguments"].strip()
            if raw.startswith("{"):
                try:
                    fn["arguments"] = json.loads(raw)
                except json.JSONDecodeError:
                    pass


def _inject_tool_thought_signatures(msg_dict: dict[str, Any]) -> None:
    if not _INJECT_TOOL_THOUGHT_SIGNATURE:
        return
    if msg_dict.get("role") != "assistant" or not msg_dict.get("tool_calls"):
        return
    for tc in msg_dict["tool_calls"]:
        if isinstance(tc, dict) and "thought_signature" not in tc:
            tc["thought_signature"] = _TOOL_THOUGHT_SIGNATURE_PLACEHOLDER


def ensure_tool_call_ids_on_assistant(msg_dict: dict[str, Any]) -> None:
    """Give each assistant tool_call a stable ``id`` so Gemini can pair tool results.

    Ollama sometimes omits ``id`` on tool_calls; without ``tool_call_id`` on the
    following tool message, Gemini may reject the request (empty function_response.name).
    """
    if msg_dict.get("role") != "assistant" or not msg_dict.get("tool_calls"):
        return
    for i, tc in enumerate(msg_dict["tool_calls"]):
        if not isinstance(tc, dict):
            continue
        if not tc.get("id"):
            tc["id"] = f"inv-{secrets.token_hex(6)}-{i}"


def normalize_ollama_chat_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Prepare chat history for Ollama /api/chat (tool rounds with Gemini-family cloud models)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d = _ollama_message_to_dict(m)
        _fix_tool_call_arguments_objects(d)
        _inject_tool_thought_signatures(d)
        out.append(d)
    return out


def assistant_response_to_stored_dict(msg: Any) -> dict[str, Any]:
    """Convert an assistant ``Message`` from chat() into a storable dict for the next turn."""
    d = _ollama_message_to_dict(msg)
    _fix_tool_call_arguments_objects(d)
    _inject_tool_thought_signatures(d)
    return d


def ollama_chat_raw_messages(
    client: Client,
    model: str,
    messages: list[dict[str, Any]],
    tools: Sequence[Any] | None = None,
) -> ChatResponse:
    """Call ``POST /api/chat`` with plain message dicts (no Pydantic round-trip).

    The ollama Python client's ``chat()`` validates messages with ``Message``,
    which rebuilds ``tool_calls`` using a ``ToolCall`` model that omits
    ``thought_signature``. Gemini 3 on Ollama Cloud requires that field in the
    JSON body; this path sends the payload unchanged.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools is not None:
        payload["tools"] = [
            t.model_dump(exclude_none=True) for t in _copy_tools(tools)
        ]
    try:
        r = client._client.post("/api/chat", json=payload)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ResponseError(e.response.text, e.response.status_code) from None
    return ChatResponse(**r.json())


def get_client(api_key: str | None = None, host: str | None = None) -> Client:
    """Return a configured Ollama Client, creating one if needed.

    When *api_key* or *host* are supplied they override the module-level
    defaults (useful for the GUI where the user enters a key at runtime).
    Passing new values resets the cached client.
    """
    global _client, OLLAMA_API_KEY, OLLAMA_HOST

    key = api_key or OLLAMA_API_KEY
    h = host or OLLAMA_HOST

    if api_key and api_key != OLLAMA_API_KEY:
        OLLAMA_API_KEY = api_key
        _client = None
    if host and host != OLLAMA_HOST:
        OLLAMA_HOST = host
        _client = None

    if _client is None:
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        _client = Client(host=h, headers=headers)

    return _client
