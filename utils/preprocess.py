import cv2

def load_and_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

