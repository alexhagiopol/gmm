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
from scipy.stats import norm


# tuning parameters
FILEPATH = "dist_1.png"
COMPONENTS = 2
ITERATIONS = 10
INIT_VARIANCE = np.float32(1/255)


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


def main():
    image = np.float32(cv2.imread(FILEPATH))
    image = image / np.float32(255)
    rows, cols, chans = image.shape
    print("Starting Gaussian Mixture Model segmentation for", FILEPATH)
    image = image[:, :, 0]
    show_image((1, 1, 1), "Input Image", image, width=15, height=10, vmin=np.min(image), vmax=np.max(image))

    # select initial means from random locations in image
    means = np.zeros(COMPONENTS)
    for i in range(COMPONENTS):
        rand_col = np.int((cols - 1) * np.random.rand(1))
        rand_row = np.int((rows - 1) * np.random.rand(1))
        means[i] = image[rand_row, rand_col]
    
    # initialize other state variables of the algorithm
    variances = np.float32(np.ones(COMPONENTS)) * INIT_VARIANCE  # initial variance is 1/255 - quite small
    stdevs = np.sqrt(variances)
    weights = np.ones(COMPONENTS)
    logLikelyhood = np.zeros(ITERATIONS)

    for i in range(ITERATIONS):
        # Expectation Step - see page 438 of Pattern Recognition and Machine Learning
        responsibilities = np.zeros((rows, cols, COMPONENTS))
        denominator = np.zeros((rows, cols))  # denominator of responsibilities equation 9.13
        for k in range(COMPONENTS):
            denominator = denominator + weights[k] * sp.stats.norm.pdf(image, means[k], stdevs[k])
        for k in range(COMPONENTS):
            responsibilities[:, :, k] = np.divide(weights[k] * sp.stats.norm.pdf(image, means[k], stdevs[k]), denominator)  # compute responsibilities eqn 9.13

        # Maximization Step - see page 439 of Pattern Recognition and Machine Learning
        numPoints = np.sum(np.sum(responsibilities, axis=0), axis=0)  # compute Nk
        for k in range(COMPONENTS):
            means[k] = np.sum(np.sum(np.multiply(responsibilities[:, :, k], image), axis=0), axis=0) / numPoints[k]
            variances[k] = np.sum(np.sum(np.multiply(responsibilities[:, :, k], image - means[k]), axis=0), axis=0) / numPoints[k]
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
        #kernel = np.ones((3, 3))
        #segmentation_output = cv2.morphologyEx(segmentation_output, cv2.MORPH_OPEN, kernel)
        show_image((1, 1, 1), "Segmentation Image " + str(i), segmentation_output, width=15, height=10, vmin=np.min(segmentation_output), vmax=np.max(segmentation_output))
        print("completed iteration", i, "with log likelyhood", logLikelyhood[i])
    plt.show()


if __name__ == "__main__":
    main()












