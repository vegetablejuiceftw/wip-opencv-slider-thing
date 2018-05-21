import cv2
from glob import glob
import numpy as np

avg = None

images = list(sorted(glob('scanned/*.jpeg')))

for fn in images:
    im = cv2.imread(fn)

    if avg is None:
        avg = np.zeros(im.shape, np.float32)

    cv2.accumulate(im, avg)

avg = avg / len(images)

res = cv2.convertScaleAbs(avg)

while True:
    cv2.imshow('img', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
