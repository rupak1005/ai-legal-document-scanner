
# utils/ocr.py
import pytesseract
import cv2

def extract_text(image):
    return pytesseract.image_to_string(image)

def draw_text_boxes(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        if int(data['conf'][i]) > 60:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image
