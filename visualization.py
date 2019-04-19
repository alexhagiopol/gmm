import image_processing
import matplotlib as mpl
mpl.use('TkAgg')  # hack around bug in matplotlib. see https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy as sp


def show_image(location, title, img, width=15, height=3, open_new_window=True, vmin=-5000.0, vmax=5000.0, cmap='gray', fontsize=10, postprocessing=False):
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
    if postprocessing:
        img = image_processing.postprocess_segmentation_images(img)
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


def visualize_algorithm_state(
        image,
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
        init_log_likelihoods):
    """
    :param image: ndarray with grayscale image
    :param responsibilities: NxMxK matrix of responsibility values as defined in equation 9.23
    :param i: current iteration index
    :param iterations: total number of iterations
    :param means_list: list of mean values, one for each component
    :param stdevs_list: list of stdev values, one for each component
    """

    segmentation_output = image_processing.compute_segmentation(image, responsibilities, means_list, stdevs_list)
    '''
    Visualizations 1 and 2 are subplots of the same figure to show the evolution of the state of the algorithm.
    '''
    plt.figure(0)
    # Visualization 1: segmentation image
    show_image((2, iterations, 1 + i),
               "Segmentation #" + str(i),
               segmentation_output,
               width=60,
               height=30,
               fontsize=14,
               vmin=np.min(segmentation_output),
               vmax=np.max(segmentation_output),
               open_new_window=False,
               postprocessing=False)

    # Visualization 2: Show Gaussian curves of each model.
    curve_points_input = np.linspace(0, 1, 1000)
    for k in range(components):
        plt.subplot(2, iterations, 1 + iterations + i)
        plt.title("Gaussian Model Curves", fontsize=10)
        plt.xlabel("Pixel Values")
        if i == 0:
            plt.ylabel("Responsibility")
        plt.ylim([0, 10])
        if means_list[k] < 0.99:  # use mean value to color the line; if the mean is too high the line will be invisible (white)
            plt.plot(curve_points_input,
                     sp.stats.norm.pdf(curve_points_input, means_list[k], stdevs_list[k]),
                     color=(means_list[k], means_list[k], means_list[k]))
        else:  # therefore if the line will be invisible, make it a dotted line instead
            plt.scatter(curve_points_input[1:1000:40],
                        sp.stats.norm.pdf(curve_points_input[1:1000:40], means_list[k], stdevs_list[k]),
                        color=(0, 0, 0), marker='.')

    '''
    Visualizations 3 and 4 created only on the final Expectation Maximization iteration to summarize the results.
    '''
    if i == iterations - 1:
        # Visualization 3: On final iteration, show un-segmented and final segmentation results on the final iteration
        plt.figure(1)
        ax = plt.subplot(3, iterations, 1 + iterations + i)
        plt.subplot(3, 2, 1)
        plt.title("Original Image", fontsize=10)
        plt.imshow(image, cmap='gray', vmin=np.min(image), vmax=np.max(segmentation_output))

        plt.subplot(3, 2, 2)
        plt.title("Final Segmented Image After " + str(iterations) + " Iterations", fontsize=10)
        postprocessed_segmentation_output = image_processing.postprocess_segmentation_images(segmentation_output)
        plt.imshow(postprocessed_segmentation_output, cmap='gray', vmin=np.min(segmentation_output), vmax=np.max(segmentation_output))

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
        plt.ylabel("Responsibility")
        for k in range(components):
            if init_means_list[k] < 0.99:  # use mean value to color the line; if the mean is too high the line will be invisible (white)
                plt.plot(curve_points_input,
                         sp.stats.norm.pdf(curve_points_input, init_means_list[k], init_stdevs_list[k]),
                         color=(init_means_list[k], init_means_list[k], init_means_list[k]))
            else:  # therefore if the line will be invisible, make it a dotted line instead
                plt.scatter(curve_points_input[1:1000:40],
                         sp.stats.norm.pdf(curve_points_input[1:1000:40], init_means_list[k], init_stdevs_list[k]),
                         color=(0, 0, 0), marker='.')

        plt.subplot(3, 2, 6)
        plt.title("Final Gaussian Mixture Curves After " + str(iterations) + " Iterations", fontsize=10)
        plt.xlabel("Pixel Values")
        for k in range(components):
            if means_list[k] < 0.99:  # use mean value to color the line; if the mean is too high the line will be invisible (white)
                plt.plot(curve_points_input,
                         sp.stats.norm.pdf(curve_points_input, means_list[k], stdevs_list[k]),
                         color=(means_list[k], means_list[k], means_list[k]))
            else:  # therefore if the line will be invisible, make it a dotted line instead
                plt.scatter(curve_points_input[1:1000:40],
                         sp.stats.norm.pdf(curve_points_input[1:1000:40], means_list[k], stdevs_list[k]),
                         color=(0, 0, 0), marker='.')

        plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=0.8)

        # Visualization 4: On final iteration, show plot of total log likelihood
        plt.figure(2)
        plt.title("Total Log Likelihood")
        plt.xlabel("Iteration")
        plt.ylabel("Total Log Likelihood")
        x = np.linspace(1, iterations, iterations)
        plt.plot(x, log_likelihoods_list[:], 'ro')
        plt.xticks(x, x)
