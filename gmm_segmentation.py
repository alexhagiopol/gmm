"""
References:
S. Russel and P. Norvig, Artificial Intelligence: A Modern Approach pp. 802 - 820 (2010)
C. Bishop, Pattern Recognition and Machine Learning pp. 423 - 459 (2011)
"""
import cv2
import matplotlib as mpl
mpl.use('TkAgg')  # hack around bug in matplotlib. see https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy as sp
import sys
import os
from scipy.stats import norm


def initialize_expectation_maximization(components, iterations):
    # initialize state variables of the algorithm
    init_variance = np.float64(10 / 255)  # initialized as explained in GMM tutorial paper
    means = np.linspace(0, 1, components)  # assume component means are evenly spaced in pixel value domain
    variances = np.float64(np.ones(components)) * init_variance  # initial variance is 10/255 - quite small
    stdevs = np.sqrt(variances)
    weights = np.ones(components)
    total_log_likelihoods = np.zeros(iterations)
    return means, variances, stdevs, weights, total_log_likelihoods


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


def visualize_algorithm_state(image, responsibilities, i, iterations, means_list, stdevs_list, total_log_likelihoods_list):
    """
    :param image: ndarray with grayscale image
    :param responsibilities: NxMxK matrix of responsibility values as defined in equation 9.23
    :param i: current iteration index
    :param iterations: total number of iterations
    :param means_list: list of mean values, one for each component
    :param stdevs_list: list of stdev values, one for each component
    """
    # create segmentation image by assigning to each pixel the mean value associated with the model with greatest prob
    rows, cols, components = responsibilities.shape
    segmentation_output = np.zeros((rows, cols))
    segmentation_output_indices = responsibilities.argmax(axis=2)
    for r in range(rows):
        for c in range(cols):
            segmentation_output[r, c] = means_list[segmentation_output_indices[r, c]]

    '''
    Visualizations 1, 2, and 3 are subplots of the same figure to show the evolution of the state of the algorithm.
    '''
    plt.figure(0)
    # Visualization 1: segmentation image
    show_image((3, iterations, 1 + i),
               "Segmentation #" + str(i),
               segmentation_output,
               vmin=np.min(segmentation_output),
               vmax=np.max(segmentation_output),
               open_new_window=False)

    # Visualization 2: Show distribution of pixel color values.
    ax = plt.subplot(3, iterations, 1 + iterations + i)
    plt.title("Pixel Value Histogram", fontsize=10)
    if i == 0:
        plt.ylabel("# Occurrences")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.hist(image.flatten(), bins=128)

    # Visualization 3: Show Gaussian curves of each model.
    curve_points_input = np.linspace(0, 1, 1000)
    for k in range(components):
        plt.subplot(3, iterations, 1 + 2*iterations + i)
        plt.title("Gaussian Mixture Curves", fontsize=10)
        plt.xlabel("Pixel Values")
        if i == 0:
            plt.ylabel("Probability")
        plt.ylim([0, 10])
        plt.plot(curve_points_input,
                 sp.stats.norm.pdf(curve_points_input, means_list[k], stdevs_list[k]),
                 color=(means_list[k], means_list[k], means_list[k]))

    '''
    Visualizations 4 and 5 created only on the final Expectation Maximization iteration to summarize the results.
    '''
    if i == iterations - 1:
        # Visualization 4: On final iteration, show un-segmented and final segmentation results on the final iteration
        plt.figure(1)
        plt.subplot(3, 2, 1)
        plt.title("Original Image", fontsize=10)
        plt.imshow(image, cmap='gray', vmin=np.min(image), vmax=np.max(segmentation_output))

        plt.subplot(3, 2, 2)
        plt.title("Final Segmented Image After " + str(iterations) + " Iterations", fontsize=10)
        plt.imshow(segmentation_output, cmap='gray', vmin=np.min(segmentation_output), vmax=np.max(segmentation_output))

        plt.subplot(3, 2, 3)
        plt.title("Pixel Value Histogram", fontsize=10)
        plt.ylabel("# Occurrences")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.hist(image.flatten(), bins=128)

        plt.subplot(3, 2, 4)
        plt.title("Pixel Value Histogram", fontsize=10)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.hist(image.flatten(), bins=128)

        plt.subplot(3, 2, 5)
        plt.title("Initial Gaussian Mixture Curves", fontsize=10)
        plt.xlabel("Pixel Values")
        plt.ylabel("Probability")
        init_means, init_variances, init_stdevs, init_weights, init_total_log_likelihoods = initialize_expectation_maximization(components, iterations)
        for k in range(components):
            plt.plot(curve_points_input,
                     sp.stats.norm.pdf(curve_points_input, init_means[k], init_stdevs[k]),
                     color=(init_means[k], init_means[k], init_means[k]))

        plt.subplot(3, 2, 6)
        plt.title("Final Gaussian Mixture Curves After " + str(iterations) + " Iterations", fontsize=10)
        plt.xlabel("Pixel Values")
        for k in range(components):
            plt.plot(curve_points_input,
                     sp.stats.norm.pdf(curve_points_input, means_list[k], stdevs_list[k]),
                     color=(means_list[k], means_list[k], means_list[k]))

        # Visualization 5: On final iteration, show plot of total log likelihood
        plt.figure(2)
        plt.title("Total Log Likelihood")
        plt.xlabel("Iteration")
        plt.ylabel("Total Log Likelihood")
        x = np.linspace(1, iterations, iterations)
        plt.plot(x, total_log_likelihoods_list[:], 'ro')
        plt.xticks(x, x)


def usage():
    print("Incorrect usage. Example:\npython gmm_segmentation.py image_filepath num_components num_iterations")


def compute_log_likelihoods(intensities, weights_list, means_list, stdevs_list):
    assert (len(weights_list) == len(means_list) == len(stdevs_list))
    K = len(weights_list)
    N, M = intensities.shape
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
    totalLogLikelihood = np.sum(np.sum(log_likelihoods_sum, axis=0), axis=0)
    return totalLogLikelihood, log_likelihoods, log_likelihoods_sum


def compute_expectation_responsibilities(intensities, weights_list, means_list, stdevs_list):
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
    totalLogLikelihood, log_likelihoods, log_likelihoods_sum = compute_log_likelihoods(intensities, weights_list, means_list, stdevs_list)
    # implement probability division as log likelihood subtraction
    for k in range(K):
        responsibilities[:, :, k] = np.exp(np.subtract(log_likelihoods[:, :, k], log_likelihoods_sum))  # responsibilities will be in linear space
    return responsibilities


def execute_segmentation(filepath, components, iterations):
    """
    :param filepath: path to input grayscale image
    :param components: number of gaussian models to fit to the image
    :param iterations: number of iterations of the expectation maximization algorithm
    :param init_variance: initial variance for each model
    """
    # 1. Initialization Step
    image = np.divide(np.float64(cv2.imread(filepath)), np.float64(255))  # read image from disk. normalize.
    rows, cols, chans = image.shape
    print("Starting Gaussian Mixture Model segmentation for", filepath)
    image = image[:, :, 0]
    means_list, variances_list, stdevs_list, weights_list, total_log_likelihoods = initialize_expectation_maximization(components, iterations)

    for i in range(iterations):
        # 2. Expectation Step - see page 438 of Pattern Recognition and Machine Learning
        responsibilities = compute_expectation_responsibilities(image, weights_list, means_list, stdevs_list)
        # 3. Inference Step - (skipped until the end for speed; see visualize_algorithm_state()).
        # 4. Maximization Step - see page 439 of Pattern Recognition and Machine Learning
        numPoints = np.sum(np.sum(responsibilities, axis=0), axis=0)  # compute Nk
        for k in range(components):
            means_list[k] = np.divide(np.sum(np.sum(np.multiply(responsibilities[:, :, k], image), axis=0), axis=0), numPoints[k])
            differences_from_mean = np.subtract(image, means_list[k])
            variances_list[k] = np.divide(np.sum(np.sum(np.multiply(responsibilities[:, :, k], np.square(differences_from_mean)), axis=0), axis=0), numPoints[k])
            stdevs_list[k] = np.sqrt(variances_list[k])
            weights_list[k] = np.divide(numPoints[k], (rows * cols))
        # 5. Log Likelihood Step - despite being done in Expectation Step, must be done again *after* maximization step
        # to accurately represent the log likelihood for a complete iteration
        total_log_likelihoods[i], dummy_1, dummy_2 = compute_log_likelihoods(image, weights_list, means_list, stdevs_list)
        # Print algorithm state.
        print("ITERATION", i,
              "\nmeans", means_list,
              " \nstdevs", stdevs_list,
              "\nweights", weights_list,
              "\nlog likelihood", total_log_likelihoods[i])
        # visualize
        visualize_algorithm_state(image, responsibilities, i, iterations, means_list, stdevs_list, total_log_likelihoods)
    plt.show()


def main():
    # process user input and exit if invalid
    if len(sys.argv) != 4:
        usage()
        exit()
    filepath = sys.argv[1]
    components = int(sys.argv[2])
    iterations = int(sys.argv[3])
    if not os.path.exists(filepath):
        usage()
        exit()
    # main segmentation function that implements GMM
    execute_segmentation(filepath, components, iterations)


if __name__ == "__main__":
    main()
