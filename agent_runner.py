"""
agent_runner.py -- builds and runs the tool-calling agent.

create_agent() (LangChain v1) wires the Groq vision model to the @tool
functions in tools.py. The MODEL decides which tool to call and loops
until the task is done; we then collect the annotated image, the
structured detection and the trace of tool calls for the UI.
"""

import base64
import io

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

import tools

SYSTEM_PROMPT = (
    "You are an object-localization agent. The user's image is attached to the first message. "
    "When the user asks where some object is, you must: "
    "1) call the find_object tool with the object name; "
    "2) if it is found, call the draw_red_box tool with the returned label, box_2d and confidence; "
    "3) if it is not found, do not draw anything and say so. "
    "Finish with one short sentence answering the user."
)


def build_agent(model_name: str, api_key: str):
    """Wire the Groq model to our two @tool functions."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=model_name, api_key=api_key, temperature=0)
    return create_agent(llm, tools=[tools.find_object, tools.draw_red_box])


def run_agent(image, query: str, model_name: str, api_key: str):
    """Run the full agent loop. Returns (output_image, detection, trace)."""

    tools.set_context(image, model_name, api_key)
    agent = build_agent(model_name, api_key)

    # Attach the image as base64 (same format as the vision specialist expects)
    working = image.copy()
    working.thumbnail((1024, 1024))
    buffered = io.BytesIO()
    working.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": f"Where is the {query}?"},
            ]
        ),
    ]

    result = agent.invoke({"messages": messages}, config={"recursion_limit": 25})

    # Build the trace: every tool call the model made + every tool result
    trace = []
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            trace.append(("call", f"{tc['name']}  args={tc['args']}"))
        if m.type == "tool":
            trace.append(("result", str(m.content)[:160]))

    output_image, detection = tools.results()
    return output_image, detection, trace
