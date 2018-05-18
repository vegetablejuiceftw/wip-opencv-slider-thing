import cv2
import numpy as np

nothing = lambda x: ()

frame = cv2.imread('color.png')
frame = cv2.resize(frame, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_CUBIC)
print(frame.shape)


# while True:
#     cv2.imshow('frame', frame)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('sliders')

# create trackbars for color change
cv2.createTrackbar('H_lower', 'sliders', 0, 255, nothing)
cv2.createTrackbar('H_upper', 'sliders', 255, 255, nothing)
cv2.createTrackbar('S_lower', 'sliders', 0, 255, nothing)
cv2.createTrackbar('S_upper', 'sliders', 255, 255, nothing)
cv2.createTrackbar('V_lower', 'sliders', 0, 255, nothing)
cv2.createTrackbar('V_upper', 'sliders', 255, 255, nothing)
blurs = [1, 3, 5, 7, 11, 15, 21, 31, 51]
cv2.createTrackbar('median_blur', 'sliders', 0, len(blurs) - 1, nothing)

while True:
    H_lower = cv2.getTrackbarPos('H_lower', 'sliders')
    H_upper = cv2.getTrackbarPos('H_upper', 'sliders')
    S_lower = cv2.getTrackbarPos('S_lower', 'sliders')
    S_upper = cv2.getTrackbarPos('S_upper', 'sliders')
    V_lower = cv2.getTrackbarPos('V_lower', 'sliders')
    V_upper = cv2.getTrackbarPos('V_upper', 'sliders')
    median_blur = cv2.getTrackbarPos('median_blur', 'sliders')

    median_frame = cv2.medianBlur(frame, blurs[median_blur])
    hsv = cv2.cvtColor(median_frame, cv2.COLOR_BGR2HSV)

    # can be considered
    # https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    # blur = cv2.bilateralFilter(img, 9, 75, 75)

    lower = np.array([H_lower, S_lower, V_lower])
    upper = np.array([H_upper, S_upper, V_upper])

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(median_frame, median_frame, mask=mask)

    cv2.imshow('median_frame', median_frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('sliders', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
