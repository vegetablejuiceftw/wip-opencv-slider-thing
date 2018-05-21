import cv2
from glob import glob

avg = None

images = list(sorted(glob('difference/*.jpeg')))

img_a = cv2.imread(images[0])
img_b = cv2.imread(images[1])

diff_color = cv2.subtract(img_a, img_b)
diff = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(diff, 25, 255, 0)

_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_b, contours, -1, (0, 255, 0), 1)

while True:
    cv2.imshow('diff', diff_color)
    cv2.imshow('img', img_b)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
