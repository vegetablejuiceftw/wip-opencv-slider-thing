# https://github.com/maitek/image_stacking
# https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
# https://github.com/spmallick/learnopencv/blob/master/ImageAlignment/image_alignment_simple_example.py

# would be interesting
# http://answers.opencv.org/question/86621/reading-geo-tiff/#86740
# https://github.com/pope/image_stacking/commit/202e593fda91483e6d692ca3eddaf3d7bbaa3c89

# python auto_stack.py "/home/toaster/PycharmProjects/wip-opencv-slider-thing/output/R_0.8 S_2017-11-01 E_2018-02-01 L_IW_VH" "output/iw-vh.jpg"

import os
import cv2
import numpy as np
from time import time


# Align and stack images with ECC method
# Slower but more accurate
def stack_images_ecc(file_list):
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    first_image = None
    stacked_image = None

    for file in sorted(file_list):
        image = cv2.imread(file, 0)
        image = image[:-100, :]

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

        median_kernel_size = 21  # 5 is ok, 25 is interesting magic number
        frame = cv2.medianBlur(frame, median_kernel_size)
        gaussian_kernel_size = 5
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
            # TODO: set criteria / iteration count
            s, warp_matrix = cv2.findTransformECC(image, first_image, warp_matrix, warp_mode)
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

    stacked_image /= len(file_list)
    stacked_image = (stacked_image * 255).astype(np.uint8)
    return stacked_image

# ===== MAIN =====
# Read all files in directory
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dir', help='Input directory of images ()')
    parser.add_argument('output_image', help='Output image name')
    parser.add_argument('--show', help='Show result image', action='store_true')
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print("ERROR {} not found!".format(image_folder))
        exit()

    file_list = os.listdir(image_folder)
    file_list = [os.path.join(image_folder, x)
                 for x in file_list if x.endswith(('.jpg', '.png', '.bmp'))]

    tic = time()

    # Stack images using ECC method
    description = "Stacking images using ECC method"
    print(description)
    result = stack_images_ecc(file_list)

    print("Stacked {} in {:.2f} seconds".format(len(file_list), (time() - tic)))

    print("Saved {}".format(args.output_image))
    cv2.imwrite(str(args.output_image), result)

    # Show image
    if args.show:
        cv2.imshow(description, result)
        cv2.waitKey(0)
