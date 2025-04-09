# --- app.py ---
import streamlit as st
from utils.preprocess import load_and_preprocess
from utils.ocr import extract_text, draw_text_boxes
from inference import analyze_layout, classify_document, draw_layout_entities, extract_entities_json, extract_key_value_pairs, generate_summary
from PIL import Image
import numpy as np
import cv2
import tempfile
import io
import json

st.set_page_config(page_title="Legal Doc Scanner", layout="centered")
st.title("AI-Powered Legal Document Scanner")

uploaded_file = st.file_uploader("Upload a scanned legal document", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    grayscale = load_and_preprocess(image)
    text = extract_text(grayscale)

# Convert grayscale to BGR before drawing boxes or displaying
    grayscale_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    image_with_boxes = draw_text_boxes(grayscale_bgr.copy())

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        entity_img = draw_layout_entities(tmp.name)
        entity_data = extract_entities_json(tmp.name)
        kv_pairs = extract_key_value_pairs(tmp.name)
        layout = analyze_layout(tmp.name)

    doc_type = classify_document(text)
    summary = generate_summary(text)

    st.image(image_with_boxes, caption="Text Detection", use_container_width=True, channels="BGR")
    st.image(entity_img, caption="Entity Labeling (LayoutLM)", use_container_width=True)
    st.subheader(f"Document Type: {doc_type}")
    st.subheader("Extracted Text")
    st.text(text)
    st.subheader("Layout Analysis")
    st.text(layout)

    st.subheader("Detected Form Fields")
    for kv in kv_pairs:
        st.markdown(f"**{kv['key']}**: {kv['value']}")

    st.subheader("Document Summary")
    st.markdown(summary)

    # Download labeled image
    buf = io.BytesIO()
    entity_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Labeled Image", data=byte_im, file_name="labeled_entity_image.png", mime="image/png")

    # Download extracted text
    st.download_button("Download Extracted Text", data=text, file_name="extracted_text.txt", mime="text/plain")

    # Download JSON entity data
    json_str = json.dumps(entity_data, indent=2)
    st.download_button("Download Key-Value Entities (JSON)", data=json_str, file_name="entities.json", mime="application/json")

    # Download key-value pairs
    kv_json = json.dumps(kv_pairs, indent=2)
    st.download_button("Download Key-Value Pairs (JSON)", data=kv_json, file_name="form_fields.json", mime="application/json")

