import cv2
from glob import glob
import numpy as np

# TODO: maybe also consider: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

avg = None

images = list(sorted(glob('difference/*.*')))

print(images)
img_a = cv2.imread(images[0])
img_b = cv2.imread(images[1])

# TODO: also bad
# shape_a, shape_b = img_a.shape, img_b.shape
#
# print(shape_a > shape_b, shape_a , shape_b)
# if shape_a > shape_b:
#     larger_image = img_a
#     smaller_image = img_b
#     blank_image = np.zeros(larger_image.shape, np.uint8)
#     h, w, *_ = smaller_image.shape
#     blank_image[:h, :w] = smaller_image
#
#     img_b = blank_image
# else:
#     larger_image = img_b
#     smaller_image = img_a
#     blank_image = np.zeros(larger_image.shape, np.uint8)
#     h, w, *_ = smaller_image.shape
#     blank_image[:h, :w] = smaller_image
#
#     img_a = blank_image
#
print(img_a.shape, img_b.shape)

# TODO: bad
h, w, *_ = img_a.shape
img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_CUBIC)

# Let's align the images
# MOTION_HOMOGRAPHY, MOTION_AFFINE
warp_mode = cv2.MOTION_HOMOGRAPHY
# 3x3 for homography
extra = int(warp_mode == cv2.MOTION_HOMOGRAPHY)
warp_matrix = np.eye(2 + extra, 3, dtype=np.float32)

subject = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
reference = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

# https://stackoverflow.com/questions/49136202/how-to-use-opencv-warpperspective-to-overlay-without-overwriting-values?rq=1
s, warp_matrix = cv2.findTransformECC(subject, reference, warp_matrix, warp_mode)
w, h, *_ = subject.shape

# Use warpPerspective for Homography
img_a = cv2.warpPerspective(img_a, warp_matrix, (h, w))

print(img_a.shape, img_b.shape)

base = "/home/microwave/PycharmProjects/untitled/difference/"
cv2.imwrite(base + "results/img_a.jpg", img_a)
cv2.imwrite(base + "results/img_b.jpg", img_b)

diff_color = cv2.subtract(img_a, img_b)
diff = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(diff, 25, 255, 0)

_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_b, contours, -1, (0, 255, 0), 1)

cv2.imwrite(base + "results/diff.jpg", diff_color)
cv2.imwrite(base + "results/outlined.jpg", img_b)

while True:
    cv2.imshow('diff', diff_color)
    cv2.imshow('img', img_b)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
