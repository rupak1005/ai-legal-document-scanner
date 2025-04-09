
# inference.py
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification, pipeline
from PIL import Image, ImageDraw
import pytesseract
import torch
import warnings

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

try:
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
    model.eval()
except Exception as e:
    warnings.warn(f"Failed to load LayoutLM model: {e}")
    tokenizer, model = None, None

def normalize_bbox(bbox, size):
    width, height = size
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

def draw_layout_entities(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    words = []
    boxes = []
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    for i in range(len(ocr_data['text'])):
        if int(ocr_data['conf'][i]) > 60:
            word = ocr_data['text'][i]
            if word.strip() == "":
                continue
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            words.append(word)
            boxes.append(normalize_bbox((x, y, x + w, y + h), (width, height)))

    if not words or not tokenizer or not model:
        return image

    with torch.no_grad():
        encoding = tokenizer(words, return_tensors="pt", truncation=True, is_split_into_words=True)
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        labels = model.config.id2label

        for word, box, pred in zip(words, boxes, predictions):
            label = labels[pred]
            if label != "O":
                x0 = int(box[0] * width / 1000)
                y0 = int(box[1] * height / 1000)
                x1 = int(box[2] * width / 1000)
                y1 = int(box[3] * height / 1000)
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                draw.text((x0, y0 - 10), label, fill="red")

    return image

def extract_entities_json(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    words = []
    boxes = []
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    for i in range(len(ocr_data['text'])):
        if int(ocr_data['conf'][i]) > 60:
            word = ocr_data['text'][i]
            if word.strip() == "":
                continue
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            words.append(word)
            boxes.append(normalize_bbox((x, y, x + w, y + h), (width, height)))

    if not words or not tokenizer or not model:
        return []

    with torch.no_grad():
        encoding = tokenizer(words, return_tensors="pt", truncation=True, is_split_into_words=True)
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        labels = model.config.id2label

    result = []
    for word, box, pred in zip(words, boxes, predictions):
        label = labels[pred]
        if label != "O":
            result.append({"text": word, "label": label, "bbox": box})

    return result

def extract_key_value_pairs(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    lines = text.split("\n")
    kv_pairs = []
    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip()
            if key and value:
                kv_pairs.append({"key": key, "value": value})
    return kv_pairs

def analyze_layout(image_path):
    return "Entity labeling visualized with LayoutLM (placeholder)."

def classify_document(text):
    if "summons" in text.lower():
        return "Summons"
    elif "notice" in text.lower():
        return "Legal Notice"
    return "Other"

def generate_summary(text):
    try:
        chunks = [text[i:i+800] for i in range(0, len(text), 800)]
        summaries = summarizer(chunks, max_length=60, min_length=25, do_sample=False)
        return "\n".join([s['summary_text'] for s in summaries])
    except Exception as e:
        return f"Summary generation failed: {e}"