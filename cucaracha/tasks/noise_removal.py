import cv2 as cv
import numpy as np


def sparse_dots(input: np.ndarray, kernel_size: int = 1):
    """Applies a median filter to remove sparse dots in the document.

    Usually, due to digitalization artifacts, there are some appearence of
    high contrast and sparse noise (also known as salt and pepper noise).

    This noise removal filter applies a median calculation though a kernel size
    defined by the user to locate these small black/white dots and correct
    then based on the neighboor values.

    Note:
        The kernel size is defines as a squared-centered area, which the values
        allowed is only odd sequence. For instance, 1, 3, 5 and so on.

    Args:
        input (np.ndarray): Input image with sparse dots noise (salt and pepper)
        kernel_size (int, optional): Kernel size in pixels. Defaults to 1.

    Raises:
        ValueError: Kernel size must be an odd value

    Returns:
        (np.ndarray, dict): Output image without major sparse dots noise. This method does not return and extra information, then get an empty dict.
    """
    if kernel_size % 2 == 0:
        raise ValueError('Kernel size must be an odd value.')

    return cv.medianBlur(input, kernel_size), {}
