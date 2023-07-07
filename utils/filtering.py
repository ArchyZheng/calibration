# %%
import numpy as np


def generate_gaussian_kernel(l: int, sig: float) -> np.array:
    """

    :param sig:
    :param l:
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def gaussian_filter(kernel: np.array, image_data: np.array) -> np.array:
    """
    using gaussian kernel, to smooth the image.

    :param kernel:
    :param image_data:
    """
