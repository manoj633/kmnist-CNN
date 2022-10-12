import cv2
import operator
import numpy as np

"""
---This program takes scanned digital images of sheets each containing 360 samples.
---Here, the images undergo
-> Initial preprocessing
-> Skew correction
-> Thresholding
-> Grid Removal
---Finally the Skew corrected, Gridless, Processed images are saved as jpeg files.
"""


def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (7, 7), 0)
    proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)
        proc = cv2.dilate(proc, kernel)
        proc = cv2.erode(proc, kernel)
    return proc


def find_corners_of_largest_polygon(img):
    contours, h = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours
    contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )  # Sort by area, descending
    polygon = contours[0]
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    top_left, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    bottom_left, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    top_right, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1)
    )
    return [
        polygon[top_left][0],
        polygon[top_right][0],
        polygon[bottom_right][0],
        polygon[bottom_left][0],
    ]


def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = (
        crop_rect[0],
        crop_rect[1],
        crop_rect[2],
        crop_rect[3],
    )
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    dst = np.array(
        [[0, 0], [1500 - 1, 0], [1500 - 1, 2400 - 1], [0, 2400 - 1]], dtype="float32"
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(1500), int(2400)))
