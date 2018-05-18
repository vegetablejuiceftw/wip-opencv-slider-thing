import cv2

nothing = lambda x: ()

# Load an color image in grayscale
frame = cv2.imread('radar.png', 0)

frame = cv2.resize(frame, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_CUBIC)
print(frame.shape)

# create trackbars for color change
cv2.namedWindow('sliders')
blurs = [1, 3, 5, 7, 11, 15, 21, 31, 51]
cv2.createTrackbar('median_blur', 'sliders', 0, len(blurs) - 1, nothing)
cv2.createTrackbar('other_blur', 'sliders', 0, len(blurs) - 1, nothing)

while True:
    median_blur = cv2.getTrackbarPos('median_blur', 'sliders')
    other_blur = cv2.getTrackbarPos('other_blur', 'sliders')

    median_frame = cv2.medianBlur(frame, blurs[median_blur])

    # can be considered
    # https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    # blur = cv2.bilateralFilter(img, 9, 75, 75)

    cv2.imshow('sliders', median_frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
