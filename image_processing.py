import visualization
import cv2
import numpy as np


def lRGB_to_XYZ(lRGB_matrix):
    """
    Convert linear RGB image in numpy form to XYZ. See https://en.wikipedia.org/wiki/SRGB.
    :param lRGB_matrix: NxNx3 numpy array
    :return: NxNx3 numpy array
    """
    rows, cols, layers = lRGB_matrix.shape
    lRGB_matrix_reshaped = np.reshape(lRGB_matrix, (rows*cols, layers))
    XYZ_matrix_reshaped = np.matmul(lRGB_matrix_reshaped, np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]))
    XYZ_matrix = np.reshape(XYZ_matrix_reshaped, (rows, cols, layers))
    return XYZ_matrix


def euclidean_distance_metric_XYZ(lrgb_image_1, lrgb_image2):
    image_1_XYZ = lRGB_to_XYZ(lrgb_image_1)
    image_2_XYZ = lRGB_to_XYZ(lrgb_image2)
    xyz_subtraction = np.subtract(image_1_XYZ, image_2_XYZ)
    distance_metric = np.sqrt(np.power(xyz_subtraction[:, :, 0], 2) + np.power(xyz_subtraction[:, :, 1], 2) + np.power(
        xyz_subtraction[:, :, 2], 2))
    return distance_metric / np.max(distance_metric)  # normalize


def preprocess_images(filepath_1, filepath_2):
    image_1 = np.divide(np.float64(cv2.imread(filepath_1)), np.float64(255))  # read image from disk. grayscale. normalize.
    matrix = image_1[:, :, 0]  # use only first matrix layer in case user provides non-grayscale input. TODO: handle more gracefully
    if filepath_2 is not None:
        image_2 = np.divide(np.float64(cv2.imread(filepath_2)), np.float64(255))  # read image from disk. normalize.
        return euclidean_distance_metric_XYZ(image_1, image_2)
    else:
        return matrix
