"""
tools.py -- the agent's tools, declared with LangChain's @tool decorator.
Each tool has ONE job; the agent (agent_runner.py) decides when to call them.
The uploaded image + credentials are placed in a shared context (set_context)
before the agent runs, so the tools can reach them.
"""

from langchain_core.tools import tool

import agent      # the vision specialist (LangChain + Groq structured call, owns DetectionBox)
import drawer     # the red-box drawing (Pillow)

_CONTEXT = {"image": None, "model": None, "key": None, "detection": None, "output": None}


def set_context(image, model_name: str, api_key: str) -> None:
    """Called once per request by the UI / runner so tools can access everything."""
    _CONTEXT.update(image=image, model=model_name, key=api_key, detection=None, output=None)


@tool
def find_object(query: str) -> str:
    """Localize ANY object the user asks about in the attached image.

    Args:
        query: the object to look for, e.g. "fire hydrant", "dog", "umbrella".

    Returns:
        A JSON string: found (bool), label (str), box_2d ([ymin, xmin, ymax, xmax]
        on a 0-1000 scale) or null, confidence (0.0-1.0).
        Call this BEFORE drawing anything.
    """
    detection = agent.find_object(_CONTEXT["model"], _CONTEXT["key"], _CONTEXT["image"], query)
    if detection.found:
        _CONTEXT["detection"] = detection
    return detection.model_dump_json()


@tool
def draw_red_box(label: str, box_2d: list[int], confidence: float) -> str:
    """Draw a red rectangle around a detected object on the attached image and save it.

    Args:
        label: object name to write on the label, e.g. "fire hydrant".
        box_2d: bounding box as [ymin, xmin, ymax, xmax], each on a 0-1000 scale.
        confidence: the confidence returned by find_object (0.0-1.0).

    Returns:
        A short confirmation string.
    """
    detection = agent.DetectionBox(found=True, label=label, box_2d=box_2d, confidence=confidence)
    _CONTEXT["detection"] = detection
    _CONTEXT["output"] = drawer.draw_box(_CONTEXT["image"], detection)
    return f"drew red box for '{label}' at {box_2d}"


def results():
    """Return (output_image_or_None, detection_or_None) after the agent has run."""
    return _CONTEXT["output"], _CONTEXT["detection"]
