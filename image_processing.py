import visualization
import copy
import cv2
import numpy as np


def lrgb_to_xyz(lrgb_matrix):
    """
    Convert normalized linear RGB image in numpy form to normalized XYZ. See https://www.cs.rit.edu/~ncs/color/t_convert.html#RGB%20to%20XYZ%20&%20XYZ%20to%20RGB.
    :param lrgb_matrix: NxNx3 numpy array
    :return: NxNx3 numpy array
    """
    rows, cols, layers = lrgb_matrix.shape
    lrgb_matrix_reshaped = np.reshape(lrgb_matrix, (rows*cols, layers))
    xyz_matrix_reshaped = np.matmul(lrgb_matrix_reshaped, np.array([[0.412453, 0.357580, 0.180423],
                                                                    [0.212671, 0.715160, 0.072169],
                                                                    [0.019334, 0.119193, 0.950227]]))
    xyz_matrix = np.reshape(xyz_matrix_reshaped, (rows, cols, layers))
    return xyz_matrix


def xyz_to_lab(xyz_matrix):
    """
    Convert normalized XYZ image in numpy form to normalized LAB. See https://www.cs.rit.edu/~ncs/color/t_convert.html#RGB%20to%20XYZ%20&%20XYZ%20to%20RGB.
    :param srgb_matrix: NxNx3 numpy array
    :return: NxNx3 numpy array
    """
    # constants for Illuminant D65 "tristimulus values of reference white"; see https://en.wikipedia.org/wiki/Illuminant_D65
    x_n = 95.047
    y_n = 100.000
    z_n = 108.883
    threshold = 0.008856

    # create matrices for LAB conversion equation constants and operations.
    # do this to avoid for loops for conditional parts of conversion implementation

    # L computation
    y_y_n_matrix = xyz_matrix[:, :, 1] / y_n
    y_y_n_matrix_over_thresh = np.int32(y_y_n_matrix > threshold)
    y_y_n_matrix_under_thresh = np.int32(y_y_n_matrix <= threshold)
    L_matrix = (116 * np.power(y_y_n_matrix, 1/3) - 16) * y_y_n_matrix_over_thresh + (903.3 * y_y_n_matrix) * y_y_n_matrix_under_thresh

    # f(t) computation
    x_x_n_matrix = xyz_matrix[:, :, 0] / x_n
    x_x_n_matrix_over_thresh = np.int32(x_x_n_matrix > threshold)
    x_x_n_matrix_under_thresh = np.int32(x_x_n_matrix <= threshold)
    y_y_n_matrix = xyz_matrix[:, :, 1] / y_n
    y_y_n_matrix_over_thresh = np.int32(y_y_n_matrix > threshold)
    y_y_n_matrix_under_thresh = np.int32(y_y_n_matrix <= threshold)
    z_z_n_matrix = xyz_matrix[:, :, 2] / z_n
    z_z_n_matrix_over_thresh = np.int32(z_z_n_matrix > threshold)
    z_z_n_matrix_under_thresh = np.int32(z_z_n_matrix <= threshold)
    f_x_x_n_matrix = np.power(x_x_n_matrix, 1/3) * x_x_n_matrix_over_thresh + (7.787 * x_x_n_matrix + 16/116) * x_x_n_matrix_under_thresh
    f_y_y_n_matrix = np.power(y_y_n_matrix, 1/3) * y_y_n_matrix_over_thresh + (7.787 * y_y_n_matrix + 16/116) * y_y_n_matrix_under_thresh
    f_z_z_n_matrix = np.power(z_z_n_matrix, 1/3) * z_z_n_matrix_over_thresh + (7.787 * z_z_n_matrix + 16/116) * z_z_n_matrix_under_thresh

    # a computation
    a_matrix = 500 * (f_x_x_n_matrix - f_y_y_n_matrix)

    # b computation
    b_matrix = 200 * (f_y_y_n_matrix - f_z_z_n_matrix)

    return np.dstack((L_matrix, a_matrix, b_matrix))  # concatenate L, a, and b into a single LAB space image


def cie76_distance_metric(lab_image_1, lab_image_2):
    """
    Compute the difference between two LAB images. See https://en.wikipedia.org/wiki/Color_difference
    :param lab_image_1: NxNx3 numpy array
    :param lab_image_2: NxNx3 numpy array
    :return: NxNx1 numpy array
    """
    difference = lab_image_2 - lab_image_1
    return np.sqrt(np.power(difference[:, :, 0], 2) + np.power(difference[:, :, 1], 2) + np.power(difference[:, :, 2], 2))


def xyz_euclidean_distance_metric(lrgb_image_1, lrgb_image2):
    image_1_XYZ = lrgb_to_xyz(lrgb_image_1)
    image_2_XYZ = lrgb_to_xyz(lrgb_image2)
    xyz_subtraction = np.subtract(image_1_XYZ, image_2_XYZ)
    distance_metric = np.sqrt(np.power(xyz_subtraction[:, :, 0], 2) + np.power(xyz_subtraction[:, :, 1], 2) + np.power(
        xyz_subtraction[:, :, 2], 2))
    return distance_metric / np.max(distance_metric)  # normalize


def preprocess_images(filepath_1, filepath_2):
    image_1 = np.divide(np.float64(cv2.imread(filepath_1)), np.float64(255))  # read image from disk. normalize.
    matrix = image_1[:, :, 0]  # use only first matrix layer in case user provides non-grayscale input. TODO: handle more gracefully
    if filepath_2 is not None:
        image_2 = np.divide(np.float64(cv2.imread(filepath_2)), np.float64(255))  # read image from disk. normalize.
        cie76_difference = cie76_distance_metric(xyz_to_lab(lrgb_to_xyz(image_1)), xyz_to_lab(lrgb_to_xyz(image_2)))
        visualization.show_image((1, 1, 1), "cie76 difference", cie76_difference, vmin=np.min(cie76_difference), vmax=np.max(cie76_difference))
        cie76_segmentation = np.int32(cie76_difference > 2.3)  # "just perceptible difference" see https://en.wikipedia.org/wiki/Color_difference
        visualization.show_image((1, 1, 1), "cie76 segmentation", cie76_segmentation, vmin=np.min(cie76_segmentation),
                                 vmax=np.max(cie76_segmentation))
        return (cie76_difference + np.min(cie76_difference)) / (np.max(cie76_difference) - np.min(cie76_difference))
    else:
        return matrix
