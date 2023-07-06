# %%
import cv2
import numpy as np


def cartesian_to_polar(image_tensor: np.array, original_point: [int, int], max_radius: int) -> np.array:
    """
    convert cartesian system into polar system, according the original point

    :param max_radius: the maximum radius of polar system, using the number of pixels to indicate the length
    :param image_tensor:
    :param original_point:
    """
    return cv2.linearPolar(image_tensor, original_point, max_radius, cv2.INTER_LINEAR)


def polar_to_cartesian(polar_tensor: np.array, original_point: [int, int], max_radius: int) -> np.array:
    """
    convert polar system into cartesian system, according the original point

    :param max_radius:
    :param image_tensor:
    :param original_point:
    """
    pass