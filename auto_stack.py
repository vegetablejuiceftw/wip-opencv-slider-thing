# https://github.com/maitek/image_stacking
# https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
# https://github.com/spmallick/learnopencv/blob/master/ImageAlignment/image_alignment_simple_example.py

# would be interesting
# http://answers.opencv.org/question/86621/reading-geo-tiff/#86740
# https://github.com/pope/image_stacking/commit/202e593fda91483e6d692ca3eddaf3d7bbaa3c89

# python auto_stack.py "/home/toaster/PycharmProjects/wip-opencv-slider-thing/output/R_0.8 S_2017-11-01 E_2018-02-01 L_IW_VH" "output/iw-vh-%s.jpg"
# python auto_stack.py "/home/toaster/PycharmProjects/wip-opencv-slider-thing/output-1/R_3.0 S_2017-11-01 E_2018-02-01 L_IW-VH-DB" "output-1/iw-vh-%s.jpg"
# python auto_stack.py "/home/microwave/PycharmProjects/untitled/output/R_0.8 S_2017-11-01 E_2018-02-01 L_IW_VH" "output/iw-vh-%s.jpg"
# python auto_stack.py "/home/microwave/PycharmProjects/untitled/output-JUN-JUL/R_1.0 S_2017-06-01 E_2018-8-01 L_IW_VH" "output-JUN-JUL/iw-vh-%s.jpg"
# python auto_stack.py "/home/microwave/PycharmProjects/untitled/output-SEP-OCT/R_1.0 S_2017-09-01 E_2018-11-01 L_IW_VH" "output-SEP-OCT/iw-vh-%s.jpg"
# python auto_stack.py "/home/microwave/PycharmProjects/untitled/imgs/asd" "imgs/iw-vh-%s.jpg"
# python auto_stack.py "/home/microwave/PycharmProjects/untitled/imgs/imgs-2017" "imgs/iw-vh-%s.jpg"

import os
import cv2
import numpy as np
from time import time

initials = {}
# Align and stack images with ECC method
# Slower but more accurate
def stack_images_ecc(file_list, base=None, median_kernel_size=17, gaussian_kernel_size=5):
    file_list = list(sorted(file_list))[:25]

    # MOTION_HOMOGRAPHY, MOTION_AFFINE
    warp_mode = cv2.MOTION_AFFINE
    # 3x3 for homography
    extra = int(warp_mode == cv2.MOTION_HOMOGRAPHY)
    warp_matrix = np.eye(2 + extra, 3, dtype=np.float32)

    # TODO: Define termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-7)

    if base is not None:
        base = base.astype(np.float32) / 255
    first_image = base
    stacked_image = base

    for i, file in enumerate(file_list):
        image = cv2.imread(file, 0)
        # image = image[:-100, :]

        if stacked_image is not None and stacked_image.shape != image.shape:
            print("Resizing", stacked_image.shape, image.shape)
            h, w = stacked_image.shape
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

        frame = image

        ##############
        # processing
        # TODO: add all other processing
        # TODO: debug possibility
        # TODO: this appreach is very slow
        # changing the order of blur, normalize, thresholding (otsu) will change the outcome
        # frame_normalized = np.zeros(frame.shape)
        # frame_normalized = cv2.normalize(frame, frame_normalized, 0, 255, cv2.NORM_MINMAX)
        # frame = frame_normalized

        # 5 is ok, 25 is interesting magic number
        if median_kernel_size:
            frame = cv2.medianBlur(frame, median_kernel_size)
        if gaussian_kernel_size:
            frame = cv2.GaussianBlur(frame, (gaussian_kernel_size, gaussian_kernel_size), 0)

        frame_normalized = np.zeros(frame.shape)
        frame_normalized = cv2.normalize(frame, frame_normalized, 0, 255, cv2.NORM_MINMAX)
        frame = frame_normalized

        # optional step, THRESH_BINARY, THRESH_TOZERO
        # TODO: 1 adaptive thresholding might not work on all scenarios perhaps
        # TODO: 2 maybe some outliers are too different from all the other pictures and should be discarded
        # _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # end of procession
        ##############

        image = frame.astype(np.float32) / 255

        print(file)
        if first_image is None:
            # convert to gray scale floating point image
            first_image = image
            stacked_image = image
        else:
            # Estimate perspective transform
            stack_reference = stacked_image / (i + bool(base is not None))
            s, warp_matrix = cv2.findTransformECC(image, stack_reference, warp_matrix, warp_mode)
            w, h = image.shape

            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                image = cv2.warpPerspective(image, warp_matrix, (h, w))
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                image = cv2.warpAffine(image, warp_matrix, (h, w))

            # Align image to first image
            # image = cv2.warpPerspective(image, warp_matrix, (h, w))
            stacked_image += image

    stacked_image /= (len(file_list) + bool(base is not None))
    stacked_image = (stacked_image * 255).astype(np.uint8)

    frame_normalized = np.zeros(stacked_image.shape)
    return cv2.normalize(stacked_image, frame_normalized, 0, 255, cv2.NORM_MINMAX)


def stack_images_ecc_iterativly(file_list, output_name):
    tic = time()
    result = None
    for index, settings in enumerate([
        dict(median_kernel_size=37, gaussian_kernel_size=17),
        dict(median_kernel_size=19, gaussian_kernel_size=9),
        dict(median_kernel_size=11, gaussian_kernel_size=5),
        dict(median_kernel_size=3, gaussian_kernel_size=3),
        dict(median_kernel_size=0, gaussian_kernel_size=0),
    ]):
        result = stack_images_ecc(file_list, base=result, **settings)

        print("Stacked {} in {:.2f} seconds".format(len(file_list), (time() - tic)))

        file_name = str(output_name)
        if "%s" in file_name:
            file_name %= index

        print("Saved {}".format(file_name))
        cv2.imwrite(file_name, result)


# ===== MAIN =====
# Read all files in directory
import argparse

if __name__ == '__main__':
    import os
    print(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dir', help='Input directory of images ()')
    parser.add_argument('output_image', help='Output image name')
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print("ERROR {} not found!".format(image_folder))
        exit()

    file_list = os.listdir(image_folder)
    file_list = [
        os.path.join(image_folder, x)
        for x in file_list if x.endswith(('.jpg', '.png', '.bmp'))
        ]

    # Stack images using ECC method
    print("Stacking images using ECC method")

    stack_images_ecc_iterativly(file_list, args.output_image)
    print('Done')
