import cv2 as cv
import numpy as np


def inplane_deskew(input: np.ndarray, max_skew=10):
    """In-plane deskew method to correct the image rotation using the plane
    orientation (bi-dimensional images)

    This method uses the global Hough Lines transform, given by OpenCV to
    generate and estimated angle to correct the image orientation to the
    zero degrees (i.e. to be parallel to the x-axis)

    Note:
        This is the usual orientation for text-formatt document. However, if your
        document presents a different text orientation, then this technique will
        not perform properly.

    Args:
        input (np.ndarray): The text-document image with a skewness
        max_skew (int, optional): Maxium angle adopted for the orientation correction. Defaults to 10.

    Returns:
        (np.ndarray, dict): The output image with inplane orientation correct. It is also give the angle value (key: `angle`)
    """
    height, width = input.shape[0], input.shape[1]

    # Create a grayscale image and denoise it
    im_gs = input
    if len(im_gs.shape) == 3:
        im_gs = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    im_gs = cv.fastNlMeansDenoising(im_gs, h=3)

    # Create an inverted B&W copy using Otsu (automatic) thresholding
    im_bw = cv.threshold(im_gs, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[
        1
    ]

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv.HoughLinesP(im_bw, 1, np.pi / 180, 50, None, 50, 10)

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = (
        np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2
    )

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew)
            < abs(angle)
            < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [
            angle for angle in angles if abs(angle) < np.deg2rad(max_skew)
        ]

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            out = cv.rotate(input, cv.ROTATE_90_CLOCKWISE)
        elif angle_deg > 0:
            out = cv.rotate(input, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Rotate the image by the residual offset
    M = cv.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    out = cv.warpAffine(
        input, M, (width, height), borderMode=cv.BORDER_REPLICATE
    )

    extra = {'angle': angle_deg}
    return out, extra
