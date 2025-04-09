import streamlit as st
from utils.preprocess import load_and_preprocess
from utils.ocr import extract_text, draw_text_boxes
from inference import (
    analyze_layout, classify_document,
    draw_layout_entities, extract_entities_json,
    extract_key_value_pairs, generate_summary
)
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import tempfile
import io
import json
import os

POPPLER_PATH = os.getenv("POPPLER_PATH", "/usr/bin")

st.set_page_config(page_title="Legal Doc Scanner", layout="centered")
st.title("📄 AI-Powered Legal Document Scanner")

uploaded_file = st.file_uploader("Upload Scanned Legal Document", type=['jpg', 'png', 'pdf'])

if uploaded_file:
    images = []

    try:
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read(), poppler_path=POPPLER_PATH)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))]

        for pil_img in images:
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            grayscale = load_and_preprocess(image)

            text = extract_text(grayscale)
            image_with_boxes = draw_text_boxes(image.copy())

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                entity_img_np = draw_layout_entities(tmp.name)

                if entity_img_np is None or not isinstance(entity_img_np, np.ndarray):
                    raise ValueError("Entity labeling failed: Output is not a valid image.")

                entity_data = extract_entities_json(tmp.name)
                kv_pairs = extract_key_value_pairs(tmp.name)

            layout = analyze_layout(text)
            doc_type = classify_document(text)
            summary = generate_summary(text)

            st.image(image_with_boxes, caption="📌 Text Detection (OCR)", use_container_width=True, channels="BGR")
            st.image(entity_img_np, caption="📌 Entity Labeling (LayoutLM)", use_container_width=True, channels="BGR")

            st.subheader(f"🗂️ Document Type: `{doc_type}`")
            st.subheader("📝 Extracted Text")
            st.text(text)

            st.subheader("🧩 Layout Analysis")
            st.text(layout)

            st.subheader("🔑 Detected Form Fields")
            for kv in kv_pairs:
                st.markdown(f"**{kv['key']}**: {kv['value']}")

            st.subheader("📃 Document Summary")
            st.markdown(summary)

            entity_pil = Image.fromarray(cv2.cvtColor(entity_img_np, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            entity_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button("Download Labeled Image", data=byte_im, file_name="labeled_entity_image.png", mime="image/png")
            st.download_button("Download Extracted Text", data=text, file_name="extracted_text.txt", mime="text/plain")
            st.download_button("Download Key-Value Entities (JSON)", data=json.dumps(entity_data, indent=2), file_name="entities.json", mime="application/json")
            st.download_button("Download Form Fields (JSON)", data=json.dumps(kv_pairs, indent=2), file_name="form_fields.json", mime="application/json")

    except Exception as e:
        st.error(f"An error occurred while processing the document: {str(e)}")