"""
References:
S. Russel and P. Norvig, Artificial Intelligence: A Modern Approach pp. 802 - 820 (2010)
C. Bishop, Pattern Recognition and Machine Learning pp. 423 - 459 (2011)
A. Hagiopol, Gaussian Mixture Models and Expectation Maximization: A Hands-On Tutorial (2019)
"""
import visualization
import image_processing

import argparse
import copy
import cProfile
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pstats
import scipy as sp
from scipy.stats import norm
import typing

class GMM_Estimator:
    def __init__(self, visualization_enabled: bool, fast_math_enabled: bool):
        self.visualization_enabled = visualization_enabled
        self.fast_math_enabled = fast_math_enabled
        self.fast_math_module = None
        if (self.fast_math_enabled):
            import accelerated_functions_build.accelerated_functions as af
            self.fast_math_module = af

    def initialize_expectation_maximization(
            self,
            filepath_1: str,
            filepath_2: str,
            components: int,
            iterations: int, 
            subtraction_threshold: float) -> \
                typing.Tuple[np.ndarray, typing.List[float], typing.List[float], typing.List[float], typing.List[float], typing.List[float]]:
        """
        perform initialization step
        :param filepath_1: path to first image (required)
        :param filepath_2: path to second image (optional)
        :param components: number of components
        :param iterations: number of iterations
        :return: means: list of initial mean values
        :return: variances: list of initial variance values
        :return: stdevs: list of initial stdev values
        :return: weights: list of initial weight values
        :return: log_likelihoods: list of initial log_likelihood values
        """
        # initialize data_matrix based on number of images provided
        image_1 = np.divide(np.float64(cv2.imread(filepath_1)), np.float64(255))  # read image from disk. normalize.
        data_matrix = image_1[:, :, 0]  # use only first matrix layer in case user provides non-grayscale input.
        data_matrix = data_matrix.astype(np.float64)
        if filepath_2 is not None:
            # given 2 images as input, data matrix is their CIE94 difference
            image_2 = np.divide(np.float64(cv2.imread(filepath_2)), np.float64(255))  # read image from disk. normalize.
            cie94_difference = \
                image_processing.cie94_distance_metric(image_processing.xyz_to_lab(image_processing.lrgb_to_xyz(image_1)),
                                                    image_processing.xyz_to_lab(image_processing.lrgb_to_xyz(image_2)))
            data_matrix = (cie94_difference + np.min(cie94_difference)) / (np.max(cie94_difference) - np.min(cie94_difference))  # normalize

        # initialize algorithm state based on parameters
        if subtraction_threshold is not None and components == 2:  # BG-FG thresholded subtraction initialization
            # initialization based on subtraction threshold (only appropriate for 2 images input and 2 components)
            print("Initializing GMM state based on thresholded subtraction.")
            cie94_segmentation_bg = np.int32(cie94_difference <= subtraction_threshold)
            cie94_segmentation_fg = np.int32(cie94_difference > subtraction_threshold)
            if (self.visualization_enabled):
                visualization.show_image((1, 1, 1), "CIE94 Segmentation w/ Threshold="+str(subtraction_threshold), cie94_segmentation_fg,
                                        vmin=np.min(cie94_segmentation_fg),
                                        vmax=np.max(cie94_segmentation_fg), postprocessing=False)
            # compute initial algorithm state
            rows = data_matrix.shape[0]
            cols = data_matrix.shape[1]
            fg_num_pixels = np.count_nonzero(cie94_segmentation_fg)
            bg_num_pixels = np.count_nonzero(cie94_segmentation_bg)
            fg_mean = np.sum(cie94_segmentation_fg * data_matrix) / fg_num_pixels
            bg_mean = np.sum(cie94_segmentation_bg * data_matrix) / bg_num_pixels
            fg_segmentation_array = np.reshape(cie94_segmentation_fg, (rows*cols))
            bg_segmentation_array = np.reshape(cie94_segmentation_bg, (rows*cols))
            data_matrix_array = np.reshape(data_matrix, (rows*cols))
            fg_pixels_array = [data_matrix_array[i] for i in range(0, data_matrix_array.shape[0]) if fg_segmentation_array[i]]
            bg_pixels_array = [data_matrix_array[i] for i in range(0, data_matrix_array.shape[0]) if bg_segmentation_array[i]]
            init_means = [bg_mean, fg_mean]
            init_variances = [np.var(bg_pixels_array), np.var(fg_pixels_array)]
            init_stdevs = np.sqrt(init_variances)
            init_weights = [np.float64(bg_num_pixels) / np.float64(bg_num_pixels + fg_num_pixels),
                            np.float64(fg_num_pixels) / np.float64(bg_num_pixels + fg_num_pixels)]

        else:  # arbitrary initialization based on even component spacing assumption
            print("Initializing GMM state based on evenly spaced component assumption.")
            init_means = [x for x in np.linspace(0, 1, components)]  # assume initial component means are evenly spaced in pixel value domain
            init_variance = np.float64(10 / 255)  # initialized as explained in GMM tutorial paper
            init_variances = [x for x in np.float64(np.ones(components)) * init_variance]  # initial variance is 10/255 - quite small
            init_stdevs = [x for x in np.sqrt(init_variances)]
            init_weights = [x for x in np.ones(components)]

        # log likelihoods do not have the concept of initialization; simply create a list to be populated by EM algorithm
        init_log_likelihoods = np.zeros(iterations)
        print("Completed Initialization")
        print("init_means=", str(init_means),
            "\ninit_variances=", str(init_variances),
            "\ninit_stdevs=", str(init_stdevs),
            "\ninit_weights=", str(init_weights))
        return data_matrix, init_means, init_variances, init_stdevs, init_weights, init_log_likelihoods


    def compute_expsum_stable(
            self,
            intensities: np.ndarray, 
            weights_list: typing.List[float], 
            means_list: typing.List[float], 
            stdevs_list: typing.List[float]) -> \
                typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        implement Equations 8 and 8 with part of Equation 10 in numerically stable expectation step derived in Hagiopol paper
        this function is used in both compute_log_likelihood_stable() and compute_expectation_responsibilities()
        :param intensities: NxM single layer matrix with 0-1 normalized np.float64 values.
        :param weights_list: list of np.float64 weight values - one for each of K components
        :param means_list: list of np.float64 mean values - one for each of K components
        :param stdevs_list: list of np.float64 stdev values - one for each of K components
        :return expsum: NxM matrix of sum of exp(P_nk - P_n_max)
        :return P: result of equation X.1
        :return P_max: result of equation X.2
        """
        K = len(weights_list)
        N, M = intensities.shape
        expsum = np.zeros((N, M), dtype=np.float64)
        P = np.zeros((N, M, K), dtype=np.float64)
        P_max = np.zeros((N, M), dtype=np.float64)

        # execute accelerated C++ implementation
        if self.fast_math_enabled:
            P_2D = np.zeros((N, M*K), dtype=np.float64)
            for k in range(0, K):
                P_2D[:, M*k : M*(k+1)] = P[:, :, k]
            self.fast_math_module.computeExpsumStable(intensities, weights_list, means_list, stdevs_list, expsum, P_2D, P_max)
            for k in range(0, K):
                P[:, :, k] = P_2D[:, M*k : M*(k+1)]
        # execute default Python implementation
        else:
            # implement Equation 8 derived in Hagiopol paper
            for k in range(K):
                P[:, :, k] = np.log(weights_list[k]) + np.log(sp.stats.norm.pdf(intensities, means_list[k], stdevs_list[k]))

            # implement Equation 9 derived in Hagiopol paper
            P_max = np.max(P, axis=2)

            # implement expsum calculation used in Equation 10 derived in Hagiopol paper
            for k in range(K):
                expsum += np.exp(P[:, :, k] - P_max)
            
        return expsum, P, P_max


    def compute_log_likelihood_stable(
            self,
            intensities: np.ndarray,
            weights_list: typing.List[float],
            means_list: typing.List[float],
            stdevs_list: typing.List[float]) -> float:
        """
        implement log likelihood calculation derived in Hagiopol paper
        :param intensities: NxM single layer matrix with 0-1 normalized np.float64 values.
        :param weights_list: list of np.float64 weight values - one for each of K components
        :param means_list: list of np.float64 mean values - one for each of K components
        :param stdevs_list: list of np.float64 stdev values - one for each of K components
        :return log_likelihood: scalar value
        """
        expsum, P, P_max = self.compute_expsum_stable(intensities, weights_list, means_list, stdevs_list)
        ln_inner_sum = P_max + np.log(expsum)  # inner sum of log likelihood equation
        return np.sum(np.sum(ln_inner_sum, axis=0), axis=0)  # outer sum of log likelihood equation


    def compute_expectation_responsibilities(
            self,
            intensities: np.ndarray,
            weights_list: typing.List[float],
            means_list: typing.List[float],
            stdevs_list: typing.List[float]) -> np.ndarray:
        """
        implement Equations 8 through 10 in numerically stable expectation step derived in Hagiopol paper
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

        # Equations 8 and 9 of Expectation step implemented in compute_expsum_stable() due to commonality with
        # log likelihood calculation
        expsum, P, P_max = self.compute_expsum_stable(intensities, weights_list, means_list, stdevs_list)

        # implement Equation 10 derived in Hagiopol paper
        ln_expsum = np.log(expsum)
        for k in range(K):
            ln_responsibilities[:, :, k] = P[:, :, k] - (P_max + ln_expsum)
        # expotentiate to convert back to real number space
        responsibilities = np.exp(ln_responsibilities)
        return responsibilities


    def execute_expectation_maximization(
                                        self,
                                        data_matrix: np.ndarray,
                                        components: int,
                                        iterations: int,
                                        init_means_list: typing.List[float],
                                        init_variances_list: typing.List[float],
                                        init_stdevs_list: typing.List[float],
                                        init_weights_list: typing.List[float],
                                        init_log_likelihoods_list: typing.List[float]) -> None:
        """
        :param matrix: numpy array with data to be segmented using Expectation Maximization. NxNx1 matrix is assumed
        :param components: number of gaussian models to fit to the image
        :param iterations: number of iterations of the expectation maximization algorithm
        :param init_variance: initial variance for each model
        """
        # 1. Initialization Step
        rows, cols = data_matrix.shape
        means_list = copy.deepcopy(init_means_list)
        variances_list = copy.deepcopy(init_variances_list)
        stdevs_list = copy.deepcopy(init_stdevs_list)
        weights_list = copy.deepcopy(init_weights_list)
        log_likelihoods_list = copy.deepcopy(init_log_likelihoods_list)
        assert (len(weights_list) == len(means_list) == len(stdevs_list) == len(variances_list))

        for i in range(iterations):
            # 2. Expectation Step - see page 438 of Pattern Recognition and Machine Learning
            responsibilities = self.compute_expectation_responsibilities(data_matrix, weights_list, means_list, stdevs_list)
            # 3. Inference Step - (skipped until the end for speed; see visualize_algorithm_state()).
            # 4. Maximization Step - see page 439 of Pattern Recognition and Machine Learning
            numPoints = np.sum(np.sum(responsibilities, axis=0), axis=0)  # compute Nk
            for k in range(components):
                means_list[k] = np.divide(np.sum(np.sum(np.multiply(responsibilities[:, :, k], data_matrix), axis=0), axis=0), numPoints[k])
                differences_from_mean = np.subtract(data_matrix, means_list[k])
                variances_list[k] = np.divide(np.sum(np.sum(np.multiply(responsibilities[:, :, k], np.square(differences_from_mean)), axis=0), axis=0), numPoints[k])
                stdevs_list[k] = np.sqrt(variances_list[k])
                weights_list[k] = np.divide(numPoints[k], (rows * cols))
            # 5. Log Likelihood Step
            log_likelihoods_list[i] = self.compute_log_likelihood_stable(data_matrix, weights_list, means_list, stdevs_list)
            # Print algorithm state.
            print("ITERATION", i,
                "\nmeans", means_list,
                "\nstdevs", stdevs_list,
                "\nweights", weights_list,
                "\nlog likelihood", log_likelihoods_list[i])
            # Visualize
            if (self.visualization_enabled):
                visualization.visualize_algorithm_state(
                    data_matrix,
                    responsibilities,
                    components,
                    i,
                    iterations,
                    means_list,
                    stdevs_list,
                    log_likelihoods_list,
                    init_means_list,
                    init_variances_list,
                    init_stdevs_list,
                    init_weights_list,
                    init_log_likelihoods_list)
        if (self.visualization_enabled):    
            plt.show()

def get_arguments() -> argparse.Namespace:
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
    parser.add_argument("--subtraction-threshold",
                        type=float,
                        default=None,
                        help="FG-BG subtraction threshold. Optional. 2.3 is just perceptible difference according to CIE")
    parser.add_argument("--visualization",
                    type=int,
                    default=1,
                    help="Toggle display of charts and graphs showing algorithm state.")
    parser.add_argument("--fast-math",
                type=int,
                default=0,
                help="Use pre-compiled math functions. Must run build_accelerated_functions.py before enabling this.")
    return parser.parse_args()


def usage() -> None:
    print("Usage help:\n"
          "Single image segmentation:"
          "python gmm_segmentation.py --first-image=example_data/beyonce.jpg --components=3 --iterations=8\n"
          "\n"
          "Image pair segmentation:"
          "python gmm_segmentation.py --first-image=example_data/image_pairs/1_background.png --second-image=example_data/image_pairs/1_foreground.png --components=2 --iterations=10\n")


def main() -> None:
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
    subtraction_threshold = args.subtraction_threshold
    visualization_enabled = bool(args.visualization)
    fast_math_enabled = bool(args.fast_math)
    if fast_math_enabled:
        import accelerated_functions_build.accelerated_functions as af
    if not os.path.exists(filepath_1):
        print("First image not found.")
        usage()
        exit()
    if filepath_2 is not None and not os.path.exists(filepath_2):
        print("Second image not found.")
        usage()
        exit()
    if components < 2:
        print("Number of components must be 2 or greater. Exiting.")
        exit()

    profiler = cProfile.Profile()
    profiler.enable()
    gmm_object = GMM_Estimator(visualization_enabled, fast_math_enabled)
    data_matrix, \
        init_means_list, \
        init_variances_list, \
        init_stdevs_list, \
        init_weights_list, \
        init_log_likelihoods_list = \
        gmm_object.initialize_expectation_maximization(filepath_1, filepath_2, components, iterations, subtraction_threshold)
    gmm_object.execute_expectation_maximization(data_matrix,
                                     components,
                                     iterations,
                                     init_means_list,
                                     init_variances_list,
                                     init_stdevs_list,
                                     init_weights_list,
                                     init_log_likelihoods_list)
    profiler.disable()
    profiler_stats = pstats.Stats(profiler)
    num_stats_display = 10
    print("EM algorithm complete. Displaying profiling statistics for top", num_stats_display, "longest running functions:")
    profiler_stats.sort_stats(pstats.SortKey.TIME).print_stats(num_stats_display)

if __name__ == "__main__":
    main()
