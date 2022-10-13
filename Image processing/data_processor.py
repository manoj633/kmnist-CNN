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


ourname = ["user1", "user2", "user3", "user4"]

for we in ourname:
    for x in range(4, 5):  # (1,13) for 12 characters
        for y in range(6, 7):  # (1,7) for 6 sheets of each character

            # ---<READ IMAGE>---#
            path = "Dataset/" + we + "/" + str(x) + str(y) + ".jpg"
            original = cv2.imread(path)

            # ---<PRE-PROCESSING>---#
            processed = pre_process_image(original)
            cv2.imwrite("processed.jpg", processed)

            # ---<FIND CORNERS FOR SKEW CORRECTION>---#
            corners = find_corners_of_largest_polygon(processed)

            # ---<SKEW CORRECTION>---#
            cropped = crop_and_warp(original, corners)
            cv2.imwrite("cropped.jpg", cropped)

            # ---<THRESHOLDING>---#
            frame = np.zeros([cropped.shape[0], cropped.shape[1], 1], dtype=np.uint8)
            frame.fill(255)
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            delta_frame = cv2.absdiff(frame, gray)
            td = cv2.threshold(delta_frame, 20, 255, cv2.THRESH_BINARY)[1]
            # cv2.imshow("mask",td)
            cv2.imwrite("td.jpg", td)

            # ---<MAKING/EXTRACTING GRID>---#
            horizontal = np.copy(td)
            vertical = np.copy(td)
            cols = horizontal.shape[1]
            horizontal_size = cols // 20
            horizontalStructure = cv2.getStructuringElement(
                cv2.MORPH_RECT, (horizontal_size, 1)
            )
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)
            cv2.imwrite("horizontal.jpg", horizontal)
            rows = vertical.shape[1]
            vertical_size = rows // 20
            vertStructure = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, vertical_size)
            )
            vertical = cv2.erode(vertical, vertStructure)
            vertical = cv2.dilate(vertical, vertStructure)
            cv2.imwrite("vertical.jpg", vertical)
            mask = cv2.bitwise_or(vertical, horizontal)
            mask = cv2.dilate(mask, None, iterations=1)

            # ---<CREATING IMAGE WITH NO GRID >---#
            cv2.imwrite("mask.jpg", mask)
            just_char = cv2.subtract(td, mask)
            cv2.erode(just_char, (5, 5), just_char, iterations=2)
            cv2.dilate(just_char, (5, 5), just_char, iterations=2)
            just_char = cv2.erode(just_char, None)
            just_char = cv2.dilate(just_char, None)

            # ---<SAVING just_char LOCALLY>---#
            cv2.imwrite(
                "G:/Project dataset/" + we + "/mine/" + str(x) + str(y) + ".jpg",
                just_char,
            )
    # cv2.imshow('mask',just_char)
    # print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
    print("Done")
cv2.waitKey(0)
cv2.destroyAllWindows()

# The final output of this code is the noiseless dataset sheets
