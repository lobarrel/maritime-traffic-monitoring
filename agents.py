"""Monitor and Investigator agents powered by gemma4:31b via Ollama."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import config
from image_processor import ImageData
from tools import (
    EXPLORATION_DIRECTIONS,
    TOOL_REGISTRY,
    findings,
    skipped_directions,
    set_anchor,
)

logger = logging.getLogger(__name__)

# Type alias for step callbacks.  The GUI (or any observer) can supply a
# callable that receives ``(event_name: str, data: dict)`` at each
# interesting point during an agent run.  When *None* the agents behave
# exactly as before — no overhead.
StepCallback = Callable[[str, dict[str, Any]], None] | None

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MonitorReport:
    per_image_analysis: list[dict]
    temporal_summary: str
    anomaly_detected: bool
    anomaly_description: str
    raw_response: str


@dataclass
class InvestigationReport:
    findings: list[dict]
    evidence_chain: str
    correlation: str
    skipped_directions: list[dict] = field(default_factory=list)
    raw_messages: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MONITOR_SYSTEM_PROMPT = """\
You are a maritime surveillance analyst examining Sentinel-2 satellite images.
Each image covers a ~30 km area at 10 m/pixel resolution. Ships appear as
bright specks against dark water, often with V-shaped wakes.

You will receive a temporal sequence of images (newest first) with their
capture dates.  For EACH image, report:
  - Approximate vessel count and locations (e.g. "cluster of ~8 vessels near
    the southern port", "2 vessels in the open water heading NE")
  - Port/anchorage activity level (high / moderate / low / none)
  - Any notable features (wakes, oil slicks, large formations)

After analysing all images individually, provide a TEMPORAL COMPARISON:
  - How has vessel count changed over the time span?
  - Have traffic patterns shifted?
  - Is there any anomaly — something unusual, unexpected, or worth
    investigating?

IMPORTANT: you are not allowed to use real historical events in your reasoning. 
You are only allowed to use the images and the anomaly description to reason about the anomaly.

You MUST end your response with a JSON block on its own line:
```json
{
  "anomaly_detected": true or false,
  "anomaly_description": "short description or empty string"
}
```
"""

INVESTIGATOR_SYSTEM_PROMPT = """\
You are a maritime anomaly investigator.  You have been alerted to an anomaly
detected in Sentinel-2 satellite imagery.  Your job is to systematically
explore the surrounding area to find events or activities that could explain
or be correlated with the detected anomaly. Remember that the detected anomaly is 
the effect, not the cause. The cause is what you are looking for. You must provide 
a single, precise explanation for the anomaly.

PLAUSIBLE CAUSES (non-exhaustive):
Satellite-detected anomalies can reflect many unrelated phenomena.  Keep an open
mind and weigh evidence against multiple hypotheses, including (for example):
  - Accidents and incidents (collisions, groundings, spills, distress patterns)
  - Blockages or congestion (channel obstructions, unusual queuing, closures)
  - Military or security-related activity (exercises, task groups, restricted
    movement patterns) — infer only from what the imagery supports
  - Benign but unusual commercial traffic (rerouting, seasonal surges)
  - Environmental or visibility artefacts (cloud shadows, blooms, slicks)
Do not assume a single cause before you have explored the area; treat these
as competing explanations until the evidence favours one.

TOOLS:
  - explore_direction(direction, distance_km (default 10), radius_km (default 5),
        max_temporal_images (optional, default several recent dates))
        Fetch AND automatically analyse satellite imagery of an adjacent area
        in a cardinal direction (N, E, S, W only).  By default this
        retrieves multiple recent cloud-free passes of the SAME patch (distinct
        dates, newest-first) so you can compare vessel patterns before vs after
        and temporally relate the observations to the anomaly.  Pass
        max_temporal_images=1 if you only need a single snapshot.  Returns
        image path(s) and a visual analysis (including a temporal synthesis
        when multiple dates are present).  You do NOT need to call
        analyze_image afterwards unless you need a follow-up question.
  - skip_direction(direction, reason)
        Explicitly record that you are NOT exploring a direction, and why.
  - analyze_image(image_path, question)
        Ask a focused follow-up question about a specific image that was
        already retrieved.  Use this only when you need to examine a
        particular detail more closely (e.g. a specific cluster of vessels
        or a port area).
  - submit_finding(title, description, evidence_images, confidence)
        Record a finding with evidence and confidence level.

INVESTIGATION STRATEGY:
1. First, reason about the anomaly and the geographic context. Carefully inspect 
   all the parts of the image where the anomaly is located (the most recent image)
   and try to find elements that could plausibly be related to the anomaly. 
2. Consider what kind of activity in each of the four cardinal directions could plausibly
   be related (shipping lanes, ports, coastline, anchorages, open ocean).
   Carefully decide which directions are WORTH exploring and which are NOT relevant.
   For each direction you skip, call skip_direction with your reasoning. Be selective 
   and strategic — explore the most likely directions first. Do not explore directions 
   that are clearly irrelevant.  Always justify your spatial reasoning.
3. For each promising direction, call explore_direction.  Read the returned
   visual analysis carefully — when multiple dates are included, use the
   temporal comparison to link cause (e.g. channel obstruction) and effect
   (e.g. upstream/downstream accumulation).  If something needs closer
   inspection, call analyze_image on one of the returned image paths.
4. When you are confident that you have found a precise explanation, call submit_finding.
   Each finding must describe:
     - WHERE the evidence was found (direction and distance from anomaly)
     - WHAT was observed (vessels, port activity, formations, etc.)
     - HOW it correlates with the detected anomaly
5. After submitting findings, stop calling tools and provide a final text
   summary that correlates ALL spatial evidence with the original anomaly.

You will receive a follow-up prompt for a formal conclusion; that response
must commit to exactly ONE primary explanation for the anomaly (see that
prompt).  Do not leave the investigation as only a list of possibilities.

IMPORTANT: you are not allowed to use real historical events in your reasoning. 
You are only allowed to use the images and the anomaly description to reason about the anomaly.
"""

# Appended when native Ollama tools are unavailable (e.g. Gemini 3 cloud — ollama#14567).
INVESTIGATOR_TEXT_TOOLS_SUFFIX = """
TOOL INVOCATION (required when native function calling is not available):
You cannot use server-side tool APIs. To run a tool, output one JSON object per tool
on its own line (valid JSON only — no markdown fences). Shape:
  {"name": "<tool_name>", "arguments": <object>}

Examples:
  {"name": "skip_direction", "arguments": {"direction": "N", "reason": "Open water only"}}
  {"name": "explore_direction", "arguments": {"direction": "E", "distance_km": 10}}
  {"name": "analyze_image", "arguments": {"image_path": "/path/from/tool/result", "question": "..."}}
  {"name": "submit_finding", "arguments": {"title": "...", "description": "...", "evidence_images": "", "confidence": "medium"}}

You may output several JSON lines in one turn if needed. After each batch, wait for
tool results in the next message, then continue. Use only the tool names listed above.
Tool results will appear as user messages beginning with ``[Tool result · <name>]``.
"""


# ---------------------------------------------------------------------------
# JSON extraction helpers (fallback for imperfect model output)
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> dict | None:
    """Try to extract a JSON object from markdown fences or bare braces."""
    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
        r"(\{[^{}]*\"anomaly_detected\"[^{}]*\})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def _parse_tool_calls_from_text(text: str) -> list[dict] | None:
    """Extract ``{"name": ..., "arguments": {...}}`` objects (supports nested args)."""
    decoder = json.JSONDecoder()
    calls: list[dict] = []
    i = 0
    while i < len(text):
        brace = text.find("{", i)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError:
            i = brace + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", {})
            if not isinstance(args, dict):
                args = {}
            calls.append({"name": obj["name"], "arguments": args})
        i = brace + end
    return calls or None


def _normalize_tool_call_for_exec(
    tc: Any,
    iteration: int,
    index: int,
) -> tuple[str, dict[str, Any], str | None]:
    """Return (tool_name, arguments, call_id).  Name comes from ``name`` or ``function.name``."""
    if isinstance(tc, dict):
        call_id = tc.get("id")
        fn = tc.get("function")
        if isinstance(fn, dict):
            name = (fn.get("name") or "").strip()
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
        else:
            name = (tc.get("name") or "").strip()
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
        if not isinstance(args, dict):
            args = {}
        name = name or "unknown_tool"
        if not call_id:
            call_id = f"inv-{iteration}-{index}"
        return name, args, call_id
    name = (tc.function.name or "").strip() or "unknown_tool"
    args = tc.function.arguments
    if not isinstance(args, dict):
        args = {}
    call_id = getattr(tc, "id", None) or f"inv-{iteration}-{index}"
    return name, args, call_id


# ---------------------------------------------------------------------------
# Monitor Agent
# ---------------------------------------------------------------------------

class MonitorAgent:
    def __init__(self, model: str = config.MODEL_NAME):
        self.model = model

    def analyse(
        self,
        images: list[ImageData],
        on_step: StepCallback = None,
    ) -> MonitorReport:
        """Run the monitor analysis on a temporal sequence of images."""

        def _emit(event: str, data: dict) -> None:
            if on_step is not None:
                on_step(event, data)

        content_parts = [
            "Analyse the following Sentinel-2 images (newest first):\n"
        ]
        image_paths = []
        for i, img in enumerate(images, 1):
            content_parts.append(
                f"Image {i}: date={img.date}, cloud_cover={img.cloud_cover}%, "
                f"id={img.item_id}"
            )
            image_paths.append(img.path)

        content_parts.append(
            "\nProvide per-image analysis, temporal comparison, and the "
            "anomaly JSON block as instructed."
        )

        messages = [
            {"role": "system", "content": MONITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n".join(content_parts),
                "images": image_paths,
            },
        ]

        _emit("monitor_start", {"image_count": len(images)})

        logger.info("MonitorAgent: sending %d images to %s", len(images), self.model)
        response = config.get_client().chat(model=self.model, messages=messages)
        text = response.message.content
        logger.debug("MonitorAgent raw response:\n%s", text)

        parsed = _extract_json_block(text)
        anomaly_detected = False
        anomaly_description = ""
        if parsed:
            anomaly_detected = bool(parsed.get("anomaly_detected", False))
            anomaly_description = parsed.get("anomaly_description", "")

        per_image = []
        for i, img in enumerate(images, 1):
            per_image.append({
                "image": i,
                "date": img.date,
                "item_id": img.item_id,
            })

        report = MonitorReport(
            per_image_analysis=per_image,
            temporal_summary=text,
            anomaly_detected=anomaly_detected,
            anomaly_description=anomaly_description,
            raw_response=text,
        )

        _emit("monitor_complete", {
            "anomaly_detected": anomaly_detected,
            "anomaly_description": anomaly_description,
            "raw_response": text,
        })

        return report


# ---------------------------------------------------------------------------
# Investigator Agent
# ---------------------------------------------------------------------------

class InvestigatorAgent:
    def __init__(
        self,
        monitor_report: MonitorReport,
        lat: float,
        lon: float,
        timestamp: str = "",
        model: str = config.MODEL_NAME,
        max_iterations: int = config.INVESTIGATOR_MAX_ITERATIONS,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.lat = lat
        self.lon = lon
        self.monitor_report = monitor_report

        findings.clear()
        skipped_directions.clear()

        set_anchor(lat, lon, timestamp, anomaly=monitor_report.anomaly_description)

        self._native_tools = config.investigator_uses_native_tools(model)
        system_content = INVESTIGATOR_SYSTEM_PROMPT
        if not self._native_tools:
            system_content = INVESTIGATOR_SYSTEM_PROMPT + "\n" + INVESTIGATOR_TEXT_TOOLS_SUFFIX
            logger.info(
                "InvestigatorAgent: using text-based tool protocol (native tools disabled for this model)"
            )

        briefing = (
            f"ANOMALY BRIEFING\n"
            f"================\n"
            f"Location: ({lat:.4f}, {lon:.4f})\n"
            f"Anomaly: {monitor_report.anomaly_description}\n\n"
            f"MONITOR REPORT\n"
            f"--------------\n"
            f"{monitor_report.temporal_summary}\n\n"
            f"AVAILABLE DIRECTIONS\n"
            f"--------------------\n"
            f"You may only use explore_direction and skip_direction with these "
            f"cardinal directions: {', '.join(EXPLORATION_DIRECTIONS)} "
            f"(N=north, E=east, S=south, W=west).\n\n"
            f"Each explore_direction call fetches imagery ~15 km away in that\n"
            f"direction.  Decide which directions are worth investigating\n"
            f"based on the anomaly type and geographic context.  Skip the\n"
            f"rest with skip_direction and explain your reasoning.\n\n"
            f"Begin your investigation."
        )

        self.messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": briefing},
        ]

    def _execute_tool(self, name: str, arguments: dict) -> str:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            logger.info("InvestigatorAgent: calling %s(%s)", name, arguments)
            result = fn(**arguments)
            return str(result)
        except Exception as e:
            logger.exception("Tool %s raised an exception", name)
            return f"Error executing {name}: {e}"

    def investigate(self, on_step: StepCallback = None) -> InvestigationReport:
        """Run the agentic investigation loop."""

        def _emit(event: str, data: dict) -> None:
            if on_step is not None:
                on_step(event, data)

        tool_functions = list(TOOL_REGISTRY.values())

        _emit("investigator_start", {
            "anomaly": self.monitor_report.anomaly_description,
            "lat": self.lat,
            "lon": self.lon,
        })

        for iteration in range(1, self.max_iterations + 1):
            logger.info("InvestigatorAgent: iteration %d/%d", iteration, self.max_iterations)

            _emit("investigator_thinking", {
                "iteration": iteration,
                "max_iterations": self.max_iterations,
            })

            tools_arg = tool_functions if self._native_tools else None
            response = config.ollama_chat_raw_messages(
                config.get_client(),
                model=self.model,
                messages=config.normalize_ollama_chat_messages(self.messages),
                tools=tools_arg,
            )

            reasoning = response.message.content or ""
            parsed_native_fallback = False
            tool_calls_list: Optional[list] = None

            if self._native_tools:
                assistant_dict = config.assistant_response_to_stored_dict(response.message)
                config.ensure_tool_call_ids_on_assistant(assistant_dict)
                self.messages.append(assistant_dict)
                tool_calls_list = assistant_dict.get("tool_calls") or []
                if not tool_calls_list and reasoning:
                    parsed = _parse_tool_calls_from_text(reasoning)
                    if parsed:
                        logger.info("Recovered %d tool call(s) from text fallback", len(parsed))
                        tool_calls_list = parsed
                        parsed_native_fallback = True
            else:
                self.messages.append({"role": "assistant", "content": reasoning})
                tool_calls_list = _parse_tool_calls_from_text(reasoning) if reasoning else None

            if reasoning:
                _emit("investigator_reasoning", {
                    "iteration": iteration,
                    "content": reasoning,
                })

            if not tool_calls_list:
                logger.info("InvestigatorAgent: no more tool calls — finishing")
                break

            for idx, tc in enumerate(tool_calls_list):
                name, args, call_id = _normalize_tool_call_for_exec(tc, iteration, idx)

                _emit("tool_call", {
                    "iteration": iteration,
                    "tool": name,
                    "arguments": args,
                })

                result = self._execute_tool(name, args)

                # Gemini pairs function_response to assistant tool_calls via tool_call_id.
                # Text-style tool protocol (or native+text fallback) avoids role=tool entirely.
                if self._native_tools and not parsed_native_fallback:
                    self.messages.append({
                        "role": "tool",
                        "content": result,
                        "name": name,
                        "tool_name": name,
                        "tool_call_id": call_id,
                    })
                else:
                    self.messages.append({
                        "role": "user",
                        "content": f"[Tool result · {name}]\n{result}",
                    })

                _emit("tool_result", {
                    "iteration": iteration,
                    "tool": name,
                    "result": result,
                })

        _emit("investigator_correlating", {})

        skip_summary = ""
        if skipped_directions:
            skip_lines = [f"  - {s['direction']}: {s['reason']}" for s in skipped_directions]
            skip_summary = (
                "\n\nDirections you skipped:\n" + "\n".join(skip_lines)
            )

        correlation_prompt = (
            "Summarise your investigation.  For each direction you explored, "
            "describe what you found and how it relates to the anomaly.  "
            "For each direction you skipped, confirm why it was not relevant.  "
            "Recall that anomalies may stem from varied causes (accidents, "
            "blockages, military or security-related activity, congestion, "
            "benign traffic shifts, environmental effects, etc.).  "
            "Provide an overall conclusion that correlates ALL spatial evidence "
            "with the original anomaly.  You MUST end with a section titled "
            "exactly:\n"
            "Primary explanation:\n"
            "followed by ONE sentence that commits to a single best-supported "
            "cause for the anomaly (not a list of alternatives)."
            + skip_summary
        )
        self.messages.append({"role": "user", "content": correlation_prompt})

        response = config.ollama_chat_raw_messages(
            config.get_client(),
            model=self.model,
            messages=config.normalize_ollama_chat_messages(self.messages),
            tools=None,
        )
        correlation_text = response.message.content
        self.messages.append(config.assistant_response_to_stored_dict(response.message))

        report = InvestigationReport(
            findings=list(findings),
            evidence_chain="\n".join(
                f"- [{f['confidence']}] {f['title']}: {f['description']}"
                for f in findings
            ),
            correlation=correlation_text,
            skipped_directions=list(skipped_directions),
            raw_messages=self.messages,
        )

        _emit("investigator_complete", {
            "findings_count": len(findings),
            "correlation": correlation_text,
        })

        return report
