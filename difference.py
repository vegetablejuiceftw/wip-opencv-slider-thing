import cv2
from glob import glob
import numpy as np

# TODO: maybe also consider: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

avg = None

images = list(sorted(glob('difference/*')))

img_a = cv2.imread(images[0])
img_b = cv2.imread(images[1])

# Let's align the images
# MOTION_HOMOGRAPHY, MOTION_AFFINE
warp_mode = cv2.MOTION_HOMOGRAPHY
# 3x3 for homography
extra = int(warp_mode == cv2.MOTION_HOMOGRAPHY)
warp_matrix = np.eye(2 + extra, 3, dtype=np.float32)

subject = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
reference = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

s, warp_matrix = cv2.findTransformECC(subject, reference, warp_matrix, warp_mode)
w, h, *_ = img_a.shape

if warp_mode == cv2.MOTION_HOMOGRAPHY:
    # Use warpPerspective for Homography
    img_a = cv2.warpPerspective(img_a, warp_matrix, (h, w))

cv2.imwrite("difference/img_a.jpg", img_a)
cv2.imwrite("difference/img_b.jpg", img_b)

diff_color = cv2.subtract(img_a, img_b)
diff = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(diff, 25, 255, 0)

_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_b, contours, -1, (0, 255, 0), 1)

cv2.imwrite("difference/diff.jpg", diff_color)
cv2.imwrite("difference/outlined.jpg", img_b)

while True:
    cv2.imshow('diff', diff_color)
    cv2.imshow('img', img_b)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
