import cv2
import numpy as np

nothing = lambda x: ()

# Load an color image in grayscale
frame = cv2.imread('radar.png', 0)

frame = cv2.resize(frame, None, fx=0.50, fy=0.50, interpolation=cv2.INTER_CUBIC)
print(frame.shape)

# create trackbars for color change
cv2.namedWindow('sliders')


class TrackBarWrapper:
    def __setattr__(self, name, range):
        start, end = range
        cv2.createTrackbar(name, 'sliders', start, end, nothing)

    def __getattribute__(self, name):
        return cv2.getTrackbarPos(name, 'sliders')

trackbar = TrackBarWrapper()
trackbar.median_blur = 0, 100
trackbar.gaussian = 0, 100

# Filter size d > 5 are very slow
trackbar.bilateral_d = 0, 19
# no effect < 10  | 150 < cartoonish
trackbar.bilateral_sigma_color = 100, 200
trackbar.bilateral_sigma_space = 100, 200

trackbar.threshold_to_zero = 0, 255
trackbar.threshold_binary = 0, 255
trackbar.threshold_trunc = 0, 255


trackbar.closing = 0, 22
trackbar.closing_iterations = 0, 22
trackbar.opening = 0, 22
trackbar.opening_iterations = 0, 22

trackbar.erode = 0, 22
trackbar.erode_iterations = 0, 22

trackbar.dilate = 0, 22
trackbar.dilate_iterations = 0, 22

trackbar.threshold_binary_final = 0, 255

while True:
    median_blur = trackbar.median_blur * 2 + 1
    gaussian = trackbar.gaussian * 2 + 1

    median_frame = frame
    median_frame = cv2.GaussianBlur(median_frame, (gaussian, gaussian), 0)
    median_frame = cv2.medianBlur(median_frame, median_blur)

    # https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    if trackbar.bilateral_d != 0:
        median_frame = cv2.bilateralFilter(
            median_frame,
            trackbar.bilateral_d,
            trackbar.bilateral_sigma_color,
            trackbar.bilateral_sigma_space,
        )

    result = median_frame
    if trackbar.threshold_to_zero:
        _, result = cv2.threshold(result, trackbar.threshold_to_zero, 255, cv2.THRESH_TOZERO)

    if trackbar.threshold_binary:
        _, result = cv2.threshold(result, trackbar.threshold_binary, 255, cv2.THRESH_BINARY)

    if trackbar.threshold_trunc:
        _, result = cv2.threshold(result, trackbar.threshold_trunc, 255, cv2.THRESH_TRUNC)

    if trackbar.closing:
        d = trackbar.closing * 2 + 1
        kernel = np.ones((d, d), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=trackbar.closing_iterations)

    if trackbar.opening:
        d = trackbar.opening * 2 + 1
        kernel = np.ones((d, d), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=trackbar.opening_iterations)

    if trackbar.erode:
        d = trackbar.erode * 2 + 1
        kernel = np.ones((d, d), np.uint8)
        result = cv2.erode(result, kernel, iterations=trackbar.erode_iterations)

    if trackbar.dilate:
        d = trackbar.dilate * 2 + 1
        kernel = np.ones((d, d), np.uint8)
        result = cv2.dilate(result, kernel, iterations=trackbar.dilate_iterations)

    if trackbar.threshold_binary_final:
        _, result = cv2.threshold(result, trackbar.threshold_binary_final, 255, cv2.THRESH_BINARY)

    cv2.imshow('sliders', result)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
