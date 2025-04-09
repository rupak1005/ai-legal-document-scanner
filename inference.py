
# inference.py
from PIL import Image, ImageDraw
import numpy as np
import cv2
import random

ENTITY_COLORS = {
    "QUESTION": (255, 0, 0),
    "ANSWER": (0, 255, 0),
    "HEADER": (0, 0, 255),
    "OTHER": (255, 255, 0),
}

def draw_layout_entities(image_path):
    try:
        pil_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(pil_image)

        mock_entities = [
            {"label": "QUESTION", "bbox": [50, 50, 300, 100]},
            {"label": "ANSWER", "bbox": [60, 110, 290, 160]},
            {"label": "HEADER", "bbox": [10, 10, 500, 40]},
        ]

        for ent in mock_entities:
            box = ent["bbox"]
            label = ent["label"]
            color = ENTITY_COLORS.get(label, (255, 255, 255))
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], box[1] - 10), label, fill=color)

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[draw_layout_entities] Error: {e}")
        return None

def extract_entities_json(image_path):
    return [{"entity": "Name", "value": "John Doe"}, {"entity": "Date", "value": "2024-01-01"}]

def extract_key_value_pairs(image_path):
    return [{"key": "Case No", "value": "123/2024"}, {"key": "Court", "value": "High Court"}]

def analyze_layout(text):
    return "Structured layout detected with multiple fields."

def classify_document(text):
    return "Summons" if "court" in text.lower() else "Other"

def generate_summary(text):
    return "This document appears to be a legal summons with structured fields like case number and court details."
