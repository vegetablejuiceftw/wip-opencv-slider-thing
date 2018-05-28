from glob import glob
from pathlib import Path
import numpy as np

import cv2
from sentinelhub import DataSource

from hub import fetch_s1_layer, COORDS_BOTTOM_RIGHT, COORDS_TOP_LEFT, fetch_truecolor_layer
from utils import zoom_bbox


def process_image(path):
    filename = Path(path).name

    # frame = cv2.imread(path, 0)
    frame = cv2.imread(path)

    gaussian = 21
    # frame = cv2.GaussianBlur(frame, (gaussian, gaussian), 0)
    frame = cv2.medianBlur(frame, gaussian)
    frame_normalized = np.zeros(frame.shape)
    frame_normalized = cv2.normalize(frame, frame_normalized, 0, 255, cv2.NORM_MINMAX)
    frame = frame_normalized

    print([path, filename])
    print(['outputs/p/' + filename])
    cv2.imwrite('outputs/p/' + filename, frame)

    return frame


def process_dir(path, output_filename):
    images = list(sorted(glob(path + '/*.png')))
    if not images:
        return

    avg = None
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
    output_dir = 'outputs/' + output_dir_name
    fetch_s1_layer(layer, output_dir=output_dir, bbox=bbox, time=time)
    fetch_truecolor_layer(time, bbox, 'outputs/TC-' + output_dir_name)

    process_dir(output_dir, 'outputs/_avg-%s.png' % output_dir_name)


def full_test(ratio, start, end):
    bbox = zoom_bbox(COORDS_TOP_LEFT, COORDS_BOTTOM_RIGHT, ratio)
    time = (start, end)

    root_dir = 'output/'
    output_dir_name = 'R_%.1f S_%s E_%s' % (ratio, start, end)

    # for layer in ['S1D-VV-VH']:
    for layer in [
        'EW_HH_DB',
        'EW_HV',
        'EW_HV_DB',
        'IW-VH-DB',
        'IW_VH',
        'IW_VV',
        'IW_VV_DB',
        'TRUE_COLOR',
    ]:
        if layer != 'TRUE_COLOR':
            data_source = DataSource.SENTINEL1_IW if 'IW' in layer else DataSource.SENTINEL1_EW
            output_dir = root_dir + output_dir_name + ' L_%s' % layer
            fetch_s1_layer(layer, output_dir=output_dir, bbox=bbox, time=time, data_source=data_source)

            print('\n', layer, "AVERAGE START")
            process_dir(output_dir, '%s AVG.png' % output_dir)

        else:
            output_dir = root_dir + output_dir_name + ' L_TC'
            fetch_truecolor_layer(time, bbox, output_dir)

            process_dir(output_dir, '%s AVG.png' % output_dir)


if __name__ == '__main__':
    full_test(0.8, '2017-11-01', '2018-02-01')
