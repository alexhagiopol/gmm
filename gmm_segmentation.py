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


def execute_segmentation(filepath, components, iterations, init_variance=np.float64(1/255)):
    """
    :param filepath: path to input grayscale image
    :param components: number of gaussian models to fit to the image
    :param iterations: number of iterations of the expectation maximization algorithm
    :param init_variance: initial variance for each model
    :return:
    """

    image = np.float64(cv2.imread(filepath))
    image = image / np.float64(255)  # normalize image
    rows, cols, chans = image.shape
    print("Starting Gaussian Mixture Model segmentation for", filepath)
    image = image[:, :, 0]
    show_image((1, 1, 1), "Input Image", image, width=15, height=10, vmin=np.min(image), vmax=np.max(image))

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
    logLikelyhood = np.zeros(iterations)

    for i in range(iterations):
        # Expectation Step - see page 438 of Pattern Recognition and Machine Learning
        responsibilities = np.zeros((rows, cols, components))
        denominator = np.zeros((rows, cols))  # denominator of responsibilities equation 9.13
        for k in range(components):
            denominator = denominator + weights[k] * sp.stats.norm.pdf(image, means[k], stdevs[k])
        for k in range(components):
            responsibilities[:, :, k] = np.divide(weights[k] * sp.stats.norm.pdf(image, means[k], stdevs[k]),
                                                  denominator)  # compute responsibilities eqn 9.13

        # Maximization Step - see page 439 of Pattern Recognition and Machine Learning
        numPoints = np.sum(np.sum(responsibilities, axis=0), axis=0)  # compute Nk
        for k in range(components):
            means[k] = np.sum(np.sum(np.multiply(responsibilities[:, :, k], image), axis=0), axis=0) / numPoints[k]
            variances[k] = np.sum(np.sum(np.multiply(responsibilities[:, :, k], image - means[k]), axis=0), axis=0) / \
                           numPoints[k]
            stdevs[k] = np.sqrt(variances[k])
            weights[k] = numPoints[k] / (rows * cols)
        # log likelyhood calculation
        logLikelyhood[i] = np.sum(np.sum(np.log(denominator), axis=0), axis=0)

        # Visualization
        segmentation_output = np.zeros((rows, cols))
        segmentation_output_indices = responsibilities.argmax(axis=2)
        for r in range(rows):
            for c in range(cols):
                segmentation_output[r, c] = means[segmentation_output_indices[r, c]]

        # morphological ops to clean up noise
        # kernel = np.ones((3, 3))
        # segmentation_output = cv2.morphologyEx(segmentation_output, cv2.MORPH_OPEN, kernel)
        show_image((1, 1, 1), "Segmentation Image " + str(i), segmentation_output, width=15, height=10,
                   vmin=np.min(segmentation_output), vmax=np.max(segmentation_output))
        print("completed iteration", i, "with log likelyhood", logLikelyhood[i])
    plt.show()


def main():
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












