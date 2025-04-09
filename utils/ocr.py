
# utils/ocr.py
import pytesseract
import cv2

def extract_text(image):
    return pytesseract.image_to_string(image)

def draw_text_boxes(image):
    boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(boxes['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image
