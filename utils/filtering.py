# %%
import cv2
import numpy as np


def generate_gaussian_kernel(l: int, sig: float) -> np.array:
    """
    see https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy.
    :param sig:
    :param l:
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def generate_one_dimension_kernel(l: int, sig: float) -> np.array:
    kernel = np.zeros(shape=(l,))
    summation = 0
    center = l // 2
    for i in range(l):
        x = i - center
        kernel[i] = np.exp(-x * x / (2 * sig * sig)) / (sig * np.square(2 * 3.141592))
        summation += kernel[i]

    return kernel / summation


def convolve2d(kernel: np.array, image_data: np.array) -> np.array:
    """
    using some kernel, to smooth the image.

    :param kernel:
    :param image_data:
    """
    return cv2.filter2D(src=image_data, kernel=kernel, ddepth=-1)


def convolve1d(kernel: np.array, curve: np.array) -> np.array:
    return np.convolve(a=kernel, v=curve, mode='same')
