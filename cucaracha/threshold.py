import cv2 as cv
import numpy as np


def otsu(input: np.ndarray):
    in_process = input
    if input.shape[2] > 1:
        in_process = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
        RuntimeWarning(
            'Input image is not gray-scaled pixel. The method converted automatically.'
        )

    thr, output = cv.threshold(
        in_process, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    extra_info = {'thr_value': thr}
    return output, extra_info


def binary_threshold(input: np.ndarray, thr: int):
    in_process = input
    if input.shape[2] > 1:
        in_process = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
        RuntimeWarning(
            'Input image is not gray-scaled pixel. The method converted automatically.'
        )

    thr, output = cv.threshold(in_process, thr, 255, cv.THRESH_BINARY)

    extra_info = {'thr_value': thr}
    return output, extra_info
