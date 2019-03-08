"""
References:
S. Russel and P. Norvig, Artificial Intelligence: A Modern Approach pp. 802 - 820 (2010)
C. Bishop, Pattern Recognition and Machine Learning pp. 423 - 459 (2011)
A. Hagiopol, Gaussian Mixture Models and Expectation Maximization: A Hands-On Tutorial (2019)
"""
import argparse
import cv2
import matplotlib as mpl
mpl.use('TkAgg')  # hack around bug in matplotlib. see https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import scipy as sp
import sys
from scipy.stats import norm


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Example implementation of Gaussian Mixture Models segmentation. "
                    "Single image input: segment the image based on grayscale intensity. "
                    "Pair of images input: segment the subtraction result. ")
    parser.add_argument("--first-image",
                        type=str,
                        default=None,
                        help="Path to image file. Must be specified.")
    parser.add_argument("--second-image",
                        type=str,
                        default=None,
                        help="Path to image file. May or may not be specified.")
    parser.add_argument("--components",
                        type=int,
                        default=None,
                        help="Number of components in the mixture of Gaussians.")
    parser.add_argument("--iterations",
                        type=int,
                        default=None,
                        help="Number of Expectation Maximization iterations.")
    return parser.parse_args()


def usage():
    print("Usage help:\n"
          "Single image segmentation:"
          "python gmm_segmentation.py --first-image=example_data/beyonce.jpg --components=3 --iterations=8\n"
          "\n"
          "Image pair segmentation:"
          "python gmm_segmentation.py --first-image=example_data/image_pairs/1_background.png --second-image=example_data/image_pairs/1_foreground.png --components=2 --iterations=10\n")


def initialize_expectation_maximization(components, iterations):
    """
    perform initialization step
    :param components: number of components
    :param iterations: number of iterations
    :return: means: list of initial mean values
    :return: variances: list of initial variance values
    :return: stdevs: list of initial stdev values
    :return: weights: list of initial weight values
    :return: log_likelihoods: list of initial log_likelihood values
    """
    # initialize state variables of the algorithm
    init_variance = np.float64(10 / 255)  # initialized as explained in GMM tutorial paper
    means = np.linspace(0, 1, components)  # assume component means are evenly spaced in pixel value domain
    variances = np.float64(np.ones(components)) * init_variance  # initial variance is 10/255 - quite small
    stdevs = np.sqrt(variances)
    weights = np.ones(components)
    log_likelihoods = np.zeros(iterations)
    return means, variances, stdevs, weights, log_likelihoods


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


def visualize_algorithm_state(image, responsibilities, i, iterations, means_list, stdevs_list, log_likelihoods_list):
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
        init_means, init_variances, init_stdevs, init_weights, init_log_likelihoods = initialize_expectation_maximization(components, iterations)
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
        plt.plot(x, log_likelihoods_list[:], 'ro')
        plt.xticks(x, x)


def compute_expsum_stable(intensities, weights_list, means_list, stdevs_list):
    """
    implement equations X.1 and X.2 with part of equation X.3 in numerically stable expectation step derived in Hagiopol paper
    this function is used in both compute_log_likelihood_stable() and compute_expectation_responsibilities()
    :param intensities: NxM single layer matrix with 0-1 normalized np.float64 values.
    :param weights_list: list of np.float64 weight values - one for each of K components
    :param means_list: list of np.float64 mean values - one for each of K components
    :param stdevs_list: list of np.float64 stdev values - one for each of K components
    :return expsum: NxM matrix of sum of exp(P_nk - P_n_max)
    :return P: result of equation X.1
    :return P_max: result of equation X.2
    """
    assert (len(weights_list) == len(means_list) == len(stdevs_list))
    K = len(weights_list)
    N, M = intensities.shape

    # implement equation X.1 derived in Hagiopol paper
    P = np.zeros((N, M, K))
    for k in range(K):
        P[:, :, k] = np.log(weights_list[k]) + np.log(sp.stats.norm.pdf(intensities, means_list[k], stdevs_list[k]))

    # implement equation X.2 derived in Hagiopol paper
    P_max = np.max(P, axis=2)

    # implement expsum calculation used in equation X.3 derived in Hagiopol paper
    expsum = np.zeros((N, M))
    for k in range(K):
        expsum += np.exp(P[:, :, k] - P_max)
    return expsum, P, P_max


def compute_log_likelihood_stable(intensities, weights_list, means_list, stdevs_list):
    """
    implement log likelihood calculation derived in Hagiopol paper
    :param intensities: NxM single layer matrix with 0-1 normalized np.float64 values.
    :param weights_list: list of np.float64 weight values - one for each of K components
    :param means_list: list of np.float64 mean values - one for each of K components
    :param stdevs_list: list of np.float64 stdev values - one for each of K components
    :return log_likelihood: scalar value
    """
    expsum, P, P_max = compute_expsum_stable(intensities, weights_list, means_list, stdevs_list)
    ln_inner_sum = P_max + np.log(expsum)  # inner sum of log likelihood equation
    return np.sum(np.sum(ln_inner_sum, axis=0), axis=0)  # outer sum of log likelihood equation


def compute_expectation_responsibilities(intensities, weights_list, means_list, stdevs_list):
    """
    implement equations X.1 through X.3 in numerically stable expectation step derived in Hagiopol paper
    :param intensities: NxM single layer matrix with 0-1 normalized np.float64 values.
    :param weights_list: list of np.float64 weight values - one for each of K components
    :param means_list: list of np.float64 mean values - one for each of K components
    :param stdevs_list: list of np.float64 stdev values - one for each of K components
    :return responsibilities: NxMxK matrix of responsibility values as defined in equation 9.23
    """

    assert (len(weights_list) == len(means_list) == len(stdevs_list))
    K = len(weights_list)
    N, M = intensities.shape
    ln_responsibilities = np.zeros((N, M, K))

    # equations X.1 and X.2 of Expectation step implemented in compute_expsum_stable() due to commonality with
    # log likelihood calculation
    expsum, P, P_max = compute_expsum_stable(intensities, weights_list, means_list, stdevs_list)

    # implement equation X.3 derived in Hagiopol paper
    ln_expsum = np.log(expsum)
    for k in range(K):
        ln_responsibilities[:, :, k] = P[:, :, k] - (P_max + ln_expsum)
    # expotentiate to convert back to real number space
    responsibilities = np.exp(ln_responsibilities)
    return responsibilities


def execute_segmentation(matrix, components, iterations):
    """
    :param matrix: numpy array with data to be segmented using Expectation Maximization
    :param components: number of gaussian models to fit to the image
    :param iterations: number of iterations of the expectation maximization algorithm
    :param init_variance: initial variance for each model
    """
    # 1. Initialization Step
    rows, cols, chans = matrix.shape
    matrix = matrix[:, :, 0]
    means_list, variances_list, stdevs_list, weights_list, log_likelihoods = initialize_expectation_maximization(components, iterations)

    for i in range(iterations):
        # 2. Expectation Step - see page 438 of Pattern Recognition and Machine Learning
        responsibilities = compute_expectation_responsibilities(matrix, weights_list, means_list, stdevs_list)
        # 3. Inference Step - (skipped until the end for speed; see visualize_algorithm_state()).
        # 4. Maximization Step - see page 439 of Pattern Recognition and Machine Learning
        numPoints = np.sum(np.sum(responsibilities, axis=0), axis=0)  # compute Nk
        for k in range(components):
            means_list[k] = np.divide(np.sum(np.sum(np.multiply(responsibilities[:, :, k], matrix), axis=0), axis=0), numPoints[k])
            differences_from_mean = np.subtract(matrix, means_list[k])
            variances_list[k] = np.divide(np.sum(np.sum(np.multiply(responsibilities[:, :, k], np.square(differences_from_mean)), axis=0), axis=0), numPoints[k])
            stdevs_list[k] = np.sqrt(variances_list[k])
            weights_list[k] = np.divide(numPoints[k], (rows * cols))
        # 5. Log Likelihood Step
        log_likelihoods[i] = compute_log_likelihood_stable(matrix, weights_list, means_list, stdevs_list)
        # Print algorithm state.
        print("ITERATION", i,
              "\nmeans", means_list,
              "\nstdevs", stdevs_list,
              "\nweights", weights_list,
              "\nlog likelihood", log_likelihoods[i])
        # Visualize
        visualize_algorithm_state(matrix, responsibilities, i, iterations, means_list, stdevs_list, log_likelihoods)
    plt.show()


def preprocess_images(filepath_1, filepath_2):
    image_1 = np.divide(np.float64(cv2.imread(filepath_1)), np.float64(255))  # read image from disk. normalize.
    if filepath_2 is not None:
        image_2 = np.divide(np.float64(cv2.imread(filepath_2)), np.float64(255))  # read image from disk. normalize.
        # TODO: convert from RGB to CIELAB
        matrix = np.abs(np.subtract(image_1, image_2))
        show_image((1, 1, 1),
                   "Subtraction",
                   matrix,
                   vmin=np.min(matrix),
                   vmax=np.max(matrix),
                   open_new_window=True)
        return matrix
    else:
        return image_1


def main():
    # process user input and exit if invalid
    args = get_arguments()
    if args.first_image is None or args.components is None or args.iterations is None:
        print("Incorrect usage.")
        usage()
        print("Exiting.")
        exit()

    filepath_1 = args.first_image
    filepath_2 = args.second_image
    components = args.components
    iterations = args.iterations
    if not os.path.exists(filepath_1):
        print("First image not found.")
        usage()
        exit()
    if filepath_2 is not None and not os.path.exists(filepath_2):
        print("Second image not found.")
        usage()
        exit()

    matrix = preprocess_images(filepath_1, filepath_2)
    # main segmentation function that implements GMM
    execute_segmentation(matrix, components, iterations)


if __name__ == "__main__":
    main()
