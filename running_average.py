import cv2
from glob import glob
import numpy as np

avg = None

for fn in sorted(glob('scanned/*.jpeg')):
    im = cv2.imread(fn)

    if avg is None:
        avg = np.float32(im)

    else:
        cv2.accumulateWeighted(im, avg, 0.1)


res = cv2.convertScaleAbs(avg)

while True:
    cv2.imshow('img', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
