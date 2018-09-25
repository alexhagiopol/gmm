"""
References:
S. Russel and P. Norvig, Artificial Intelligence: A Modern Approach pp. 802 - 820 (2010)
C. Bishop, Pattern Recognition and Machine Learning pp. 423 - 459 (2011)
"""
import cv2
import matplotlib as mpl
mpl.use('TkAgg')  # hack around bug in matplotlib. see https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import sys
import os
from scipy.stats import norm


def show_image(location, title, img, width=15, height=3, open_new_window=True, vmin=-5000.0, vmax=5000.0, cmap='gray', fontsize=10):
    """
    Displays an image in a multi-image display window
    :param location: (r,c,n) tuple where r is the # of display rows, c is the # of display cols, and n is the position for img
    :param title: string with title
    :param img: ndarray with image
    :param width: int for width
    :param open_new_window: boolean true if you have not already created a new plt figure
    :param vmin: float min value to display in single layer image
    :param vmax: float max value to display in single layer image
    :param cmap: colormap for display of single layer images
    :
    """
    if open_new_window:
        plt.figure(figsize=(height, width))
    plt.subplot(*location)
    plt.title(title, fontsize=fontsize)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    if open_new_window:
        plt.show()
        plt.close()


def usage():
    print("Incorrect usage. Example:\npython gmm_segmentation.py image_filepath num_components num_iterations")


def compute_responsibilities(intensities, weights_list, means_list, stdevs_list):
    """
    implement equation 9.23 in expectation step
    :param intensities: NxM single layer matrix with 0-1 normalized np.float64 values.
    :param weights_list: list of np.float64 weight values - one for each of K components
    :param means_list: list of np.float64 mean values - one for each of K components
    :param stdevs_list: list of np.float64 stdev values - one for each of K components
    :return responsibilities: NxMxK matrix of responsibility values as defined in equation 9.23
    :return totalLogLikelihood: log likelihood value for the state of the estimator at this iteration
    """

    assert (len(weights_list) == len(means_list) == len(stdevs_list))
    K = len(weights_list)
    N, M = intensities.shape

    responsibilities = np.zeros((N, M, K))
    log_likelihoods = np.zeros((N, M, K))

    # numerator of equation 9.23 in log form for numerical stability
    # compute NxMxK matrix with likelihood of each pixel belonging to each of K components
    for k in range(K):
        log_likelihoods[:, :, k] = np.log(
            np.multiply(weights_list[k], sp.stats.norm.pdf(intensities, means_list[k], stdevs_list[k])))
    # denominator of equation 9.23
    log_likelihoods_max = np.max(log_likelihoods, axis=2)
    # implement LogSumExp technique for probability summation
    expsum = np.zeros((N, M))
    for k in range(K):
        expsum += np.exp(log_likelihoods[:, :, k] - log_likelihoods_max)
    log_likelihoods_sum = log_likelihoods_max + np.log(expsum)
    # implement probability division as log likelihood subtraction
    for k in range(K):
        responsibilities[:, :, k] = np.exp(np.subtract(log_likelihoods[:, :, k], log_likelihoods_sum))  # responsibilities will be in linear space
    # total log likelihood
    totalLogLikelihood = np.sum(np.sum(log_likelihoods_sum, axis=0), axis=0)
    return responsibilities, totalLogLikelihood


def execute_segmentation(filepath, components, iterations, init_variance=np.float64(10/255)):
    """
    :param filepath: path to input grayscale image
    :param components: number of gaussian models to fit to the image
    :param iterations: number of iterations of the expectation maximization algorithm
    :param init_variance: initial variance for each model
    """
    image = np.divide(np.float64(cv2.imread(filepath)), np.float64(255))  # read image from disk. normalize.
    rows, cols, chans = image.shape
    print("Starting Gaussian Mixture Model segmentation for", filepath)
    image = image[:, :, 0]

    # warning: random initialization means results will be different every time
    # select initial means from random locations in image
    means = np.zeros(components)
    for i in range(components):
        rand_col = np.int((cols - 1) * np.random.rand(1))
        rand_row = np.int((rows - 1) * np.random.rand(1))
        means[i] = image[rand_row, rand_col]

    # initialize other state variables of the algorithm
    variances = np.float64(np.ones(components)) * init_variance  # initial variance is 1/255 - quite small
    stdevs = np.sqrt(variances)
    weights = np.ones(components)
    totalLogLikelihoods = np.zeros(iterations)

    for i in range(iterations):
        # Expectation Step - see page 438 of Pattern Recognition and Machine Learning
        responsibilities, totalLogLikelihoods[i] = compute_responsibilities(image, weights, means, stdevs)
        # Maximization Step - see page 439 of Pattern Recognition and Machine Learning
        numPoints = np.sum(np.sum(responsibilities, axis=0), axis=0)  # compute Nk
        for k in range(components):
            means[k] = np.divide(np.sum(np.sum(np.multiply(responsibilities[:, :, k], image), axis=0), axis=0), numPoints[k])
            differences_from_mean = np.subtract(image, means[k])
            variances[k] = np.divide(np.sum(np.sum(np.multiply(responsibilities[:, :, k], np.square(differences_from_mean)), axis=0), axis=0), numPoints[k])
            stdevs[k] = np.sqrt(variances[k])
            weights[k] = np.divide(numPoints[k], (rows * cols))

        # Visualization
        segmentation_output = np.zeros((rows, cols))
        segmentation_output_indices = responsibilities.argmax(axis=2)
        for r in range(rows):
            for c in range(cols):
                segmentation_output[r, c] = means[segmentation_output_indices[r, c]]
        show_image((1 + iterations // 4, 4, i + 1), "Segmentation Image " + str(i), segmentation_output, width=15, height=10,
                   vmin=np.min(segmentation_output), vmax=np.max(segmentation_output), open_new_window=False)
        print("ITERATION", i, "\nmeans", means, " \nstdevs", stdevs, "\nweights", weights, "\nlog likelihood", totalLogLikelihoods[i])
    plt.show()
    plt.close()


def main():
    print("WARNING: Due to random initialization, every run will produce slightly different results.")
    if len(sys.argv) != 4:
        usage()
        exit()
    filepath = sys.argv[1]
    components = int(sys.argv[2])
    iterations = int(sys.argv[3])
    if not os.path.exists(filepath):
        usage()
        exit()
    execute_segmentation(filepath, components, iterations)

if __name__ == "__main__":
    main()












