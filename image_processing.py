import visualization
import cv2
import numpy as np


def preprocess_images(filepath_1, filepath_2):
    image_1 = np.divide(np.float64(cv2.imread(filepath_1)), np.float64(255))  # read image from disk. grayscale. normalize.
    if filepath_2 is not None:
        image_2 = np.divide(np.float64(cv2.imread(filepath_2)), np.float64(255))  # read image from disk. normalize.
        # TODO: convert from sRGB to CIELAB
        matrix = np.abs(np.subtract(image_1, image_2))
        visualization.show_image((1, 1, 1),
                   "Subtraction",
                   matrix,
                   vmin=np.min(matrix),
                   vmax=np.max(matrix),
                   open_new_window=True)
        return matrix
    else:
        return image_1
