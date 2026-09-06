# 🔎 Object Finder Agent (Assignment)

Give the agent **any picture** and ask it to find **anything** — a fire hydrant, a dog,
an umbrella, a bicycle… The agent returns the same picture with a **red box** drawn
around the object you asked for.

**Stack:** Streamlit (UI) · LangChain (agent) · Groq free vision API (Llama-4 / Qwen-VL) · Pillow (drawing)

**100% free** — Groq's API has a generous free tier, no credit card needed.

## How it works
1. You upload an image and type ANY object name (`app.py`).
2. `app.py` calls `run_agent()` — a LangChain tool-calling agent (`create_agent`) on a Groq vision model,
   with the image and your question attached.
3. The **model itself decides** to call the two tools (declared with `@tool` inside `agent.py`):
   - `find_object(query)` → one structured Groq vision call that localizes ANY object
   - `draw_red_box(label, box_2d, confidence)` → runs `drawer.py`, which draws the red rectangle
4. The agent answers; the UI shows Before/After, the trace of tool calls, and the raw JSON.

## Project structure (3 simple files)
```
app.py      # Streamlit web interface (thin UI only)
agent.py    # the @tools + the tool-calling agent loop
drawer.py   # drawing: red box + label on the image
```
The agent pattern is real: the model chooses which `@tool` to call, and you
can watch every call in the UI's "Agent trace" panel.

## Run it
```bash
pip install -r requirements.txt

# put your key in .env (see .env.example), or just type it in the sidebar
streamlit run app.py
```

Get a **free** Groq API key at <https://console.groq.com/keys>.

## Optional: batch-demo the pipeline offline
```bash
python make_demo.py samples/dog.jpg "dog" samples/umbrella.jpg "umbrella"
```
Any images, any queries — it draws the red boxes and writes `samples/demo_results.json`.
(Downloads a one-time open-vocabulary model on first run.)

## Deploy (also free)
Streamlit Community Cloud: push to GitHub → share.streamlit.io → New app →
main file `app.py` → Settings → Secrets → add `GROQ_API_KEY = "your-key"`.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface (thin UI only) |
| `agent.py` | The @tools (`find_object`, `draw_red_box`) + the tool-calling agent loop |
| `drawer.py` | Draws the red box + label on the image (Pillow) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for the API key |
| `samples/*.jpg` | Example inputs (fire hydrant, dog, umbrella) |
| `samples/*_detected.jpg` | Example outputs (red box drawn) |
| `samples/demo_results.json` | Run log of the example batch |
| `report/Technical_Report.docx` | Full technical report with before/after demos |
| `report/Code_Explanation_Guide.pdf` | Every code chunk explained |
