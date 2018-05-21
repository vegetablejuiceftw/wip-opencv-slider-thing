import sys
from glob import glob
from pathlib import Path
import numpy as np

import cv2

from hub import fetch_s1_layer, COORDS_BOTTOM_RIGHT, COORDS_TOP_LEFT, fetch_truecolor_layer
from utils import zoom_bbox


def process_image(path):
    filename = Path(path).name
    print(path, filename)
    # return

    frame = cv2.imread(path, 0)

    gaussian = 21
    # frame = cv2.GaussianBlur(frame, (gaussian, gaussian), 0)
    frame = cv2.medianBlur(frame, gaussian)
    frame_normalized = np.zeros(frame.shape)
    frame_normalized = cv2.normalize(frame, frame_normalized, 0, 255, cv2.NORM_MINMAX)
    frame = frame_normalized

    cv2.imwrite('outputs/p/' + filename, frame)

    return frame


def process_dir(path, output_filename):
    avg = None
    images = list(sorted(glob(path + '/*.png')))

    for image in images:
        im = process_image(image)

        if avg is None:
            avg = np.zeros(im.shape, np.float32)

        cv2.accumulate(im, avg)

    avg = avg / len(images)

    res = cv2.convertScaleAbs(avg)

    cv2.imwrite(output_filename, res)


def process_data(ratio=1.0, *, time=None, extra_suffix=None):
    bbox = zoom_bbox(COORDS_TOP_LEFT, COORDS_BOTTOM_RIGHT, ratio)

    # layer = 'S1-VH'
    # layer = 'S1D-VH-VV-VH'
    layer = 'S1D-VV-VH'
    output_dir_name = '%s__bb%.1f' % (layer, ratio)
    if extra_suffix is not None:
        output_dir_name = output_dir_name + '_' + extra_suffix
    output_dir = 'outputs/' +  output_dir_name
    fetch_s1_layer(layer, output_dir=output_dir, bbox=bbox, time=time)
    fetch_truecolor_layer(time, bbox, 'outputs/TC-' + output_dir_name)

    process_dir(output_dir, 'outputs/_avg-%s.png' % output_dir_name)


if __name__ == '__main__':
    # process_dir(sys.argv[1])
    # process_data(0.5)
    # process_data(1.0)
    # process_data(2.0)

    process_data(1.0, time=('2017-07-01', '2017-08-01'), extra_suffix='201707')
    process_data(1.0, time=('2017-11-01', '2017-12-01'), extra_suffix='201711')
    process_data(1.0, time=('2018-04-01', '2018-05-01'), extra_suffix='201804')
