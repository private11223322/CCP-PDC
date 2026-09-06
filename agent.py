"""
agent.py -- the AI agent.
A LangChain tool-calling agent with two @tools that the MODEL decides to call:
    find_object  -> locates ANY object (one Groq vision call, structured answer)
    draw_red_box -> draws the red rectangle (uses drawer.py)
One entry point for the UI: run_agent(image, query, model_name, api_key).
"""

import base64
import io

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel

import drawer

SYSTEM = (
    "You are an object-localization agent. Call find_object for the user's object. "
    "If it is found, call draw_red_box with its answer. If not found, just say so. "
    "Then answer in one short sentence."
)


class DetectionBox(BaseModel):
    """The structured answer of the vision call."""

    found: bool = False
    label: str = ""
    box_2d: list[int] | None = None  # [ymin, xmin, ymax, xmax], each 0-1000
    confidence: float = 0.0


def _to_b64(image) -> str:
    """Shrink the image and encode it as base64 text for the API."""
    small = image.copy()
    small.thumbnail((1024, 1024))
    buf = io.BytesIO()
    small.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _vision_find(model_name, api_key, image, query) -> DetectionBox:
    """One Groq vision call: picture + question -> DetectionBox."""
    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_tokens=400,           # free tier allows ~1000 output tokens/minute
        reasoning_effort="none",  # skip Qwen's hidden 'thinking' output (saves tokens)
    )
    structured = llm.with_structured_output(DetectionBox, method="json_mode")

    prompt = (
        f"You are an object-localization agent. Locate '{query}' in the image. "
        "Reply ONLY with JSON: found (bool), label (str), "
        "box_2d ([ymin, xmin, ymax, xmax], each 0-1000, or null), confidence (0-1)."
    )
    msg = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(image)}"}},
            {"type": "text", "text": prompt},
        ]
    )
    return structured.invoke([msg])


def run_agent(image, query, model_name, api_key):
    """Run the tool-calling agent. Returns (output_image, detection, trace)."""
    result = {}  # the draw tool leaves the annotated image + detection here

    @tool
    def find_object(query: str) -> str:
        """Localize ANY object in the attached image. Returns JSON:
        found, label, box_2d ([ymin, xmin, ymax, xmax], 0-1000) or null, confidence."""
        return _vision_find(model_name, api_key, image, query).model_dump_json()

    @tool
    def draw_red_box(label: str, box_2d: list[int], confidence: float) -> str:
        """Draw the red box for a found object on the image."""
        det = DetectionBox(found=True, label=label, box_2d=box_2d, confidence=confidence)
        result["detection"] = det
        result["image"] = drawer.draw_box(image, det)
        return f"red box drawn for '{label}'"

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_tokens=400,           # keep every agent step under the free-tier limit
        reasoning_effort="none",
    )
    agent = create_agent(llm, tools=[find_object, draw_red_box])

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_to_b64(image)}"}},
                {"type": "text", "text": f"Where is the {query}?"},
            ]
        ),
    ]
    steps = agent.invoke({"messages": messages}, config={"recursion_limit": 25})

    trace = []
    for m in steps["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            trace.append(f"call  {tc['name']}({tc['args']})")
        if m.type == "tool":
            trace.append(f"result  {str(m.content)[:100]}")

    return result.get("image"), result.get("detection"), trace
