import cv2 as cv
import numpy as np


def load_document(doc_path: str):
    return cv.imread(doc_path)
