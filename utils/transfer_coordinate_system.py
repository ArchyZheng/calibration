# %%
import cv2
import numpy as np
import math


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

    :param polar_tensor:
    :param max_radius:
    :param original_point:
    """
    return cv2.linearPolar(polar_tensor, original_point, max_radius, cv2.WARP_INVERSE_MAP)


def get_high_value_point(image_picture: np.array, threshold: float) -> list:
    """
    this function will return the index of the point which is higher than threshold
    :param image_picture:
    :param threshold:
    """
    row, col = np.where(image_picture > threshold)
    candidate = []
    for index in range(len(row)):
        candidate.append([row[index], col[index]])

    return candidate


def polar_vector_to_cartesian_vector(polar_vector, max_radius: int) -> list:
    """
    :TODO This function has some faults
    :param polar_vector:
    :param max_radius:
    """
    cartesian_vector = []
    for theta, pho in polar_vector:
        cartesian_vector.append(cv2.polarToCart(magnitude=pho, angle=theta, angleInDegrees=1))

    return cartesian_vector


def get_location_of_cartesian(polar_theta: float, polar_radius: float, polar_center: (int, int), width: int):
    polar_theta = 360 / width * polar_theta
    polar_theta = np.radians(polar_theta)
    return polar_center[0] + polar_radius * np.cos(polar_theta), polar_center[1] + polar_radius * np.sin(polar_theta)

def get_y_on_the_circle(x: float, radius: float, center: (int, int)) -> list:
    """
    Get the y-coordinate by substituting the x-coordinate of the circle.
    :param x:
    :param radius:
    :param center: [center_x, center_y] the location is correlated by matrix
    """
    y_without_bais = np.sqrt(radius ** 2 - (x - center[0]) ** 2)
    y_1 = y_without_bais + center[1]
    return [y_1, 2 * center[1] - y_1]


def draw_a_circle(image: np.array, center: (int, int), radius: float):
    """
    Draw a circle by modify the numpy array.
    :param image:
    :param center:
    :param radius:
    """
    for angle in np.arange(0, 360, 0.01):
        radiant = np.radians(angle)
        x = center[0] + radius * np.cos(radiant)
        y = center[1] + radius * np.sin(radiant)
        image[int(x), int(y)] = 200
