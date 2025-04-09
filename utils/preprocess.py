
# --- utils/preprocess.py ---
import cv2

def load_and_preprocess(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale
