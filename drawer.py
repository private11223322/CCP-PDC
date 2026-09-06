"""
drawer.py -- drawing.
One job: take the agent's DetectionBox and draw it on the image in red.
Pure logic, no UI. Used by app.py.
"""

from PIL import Image, ImageDraw, ImageFont


def draw_box(image: Image.Image, detection) -> Image.Image:
    """Convert the 0-1000 box to pixels and draw a red rectangle + label."""

    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    ymin, xmin, ymax, xmax = detection.box_2d  # model returns 0-1000 scale
    left = max(0, min(int(xmin / 1000 * w), w))
    top = max(0, min(int(ymin / 1000 * h), h))
    right = max(0, min(int(xmax / 1000 * w), w))
    bottom = max(0, min(int(ymax / 1000 * h), h))

    draw.rectangle([left, top, right, bottom], outline="red", width=max(3, w // 250))

    # Small red label above the box
    text = f"{detection.label} ({detection.confidence:.0%})"
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, w // 40))
    except OSError:
        font = ImageFont.load_default()
    text_bg = draw.textbbox((left, top), text, font=font)
    text_h = text_bg[3] - text_bg[1]
    label_y = top - text_h - 8 if top - text_h - 8 > 0 else top + 4
    draw.rectangle([left, label_y - 4, left + text_bg[2] - text_bg[0] + 8, label_y + text_h + 4], fill="red")
    draw.text((left + 4, label_y), text, fill="white", font=font)

    return out
