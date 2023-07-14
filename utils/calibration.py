import numpy as np
import cv2


def ellipse_curve(center: (float, float), axes: (float, float), angle: float, shape: (int, int),
                  precision: float) -> dict:
    """

    :param precision: float
    :param shape: the output shape of map
    :param center:
    :param axes: axes[0] long axes, axes[1] short axes
    :param angle:
    """
    center = center
    axes = axes
    angle = angle
    angle_radians = np.radians(angle)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    output = {}
    for theta in np.arange(0, 360, 0.1):
        theta_radians = np.radians(theta)
        cos_theta = np.cos(theta_radians)
        sin_theta = np.sin(theta_radians)

        x = center[0] + axes[0] * cos_angle * cos_theta - axes[1] * sin_theta * sin_angle
        y = center[0] + axes[0] * sin_angle * cos_theta + axes[1] * sin_theta * cos_angle

        output[x] = y

    return output


def sparse_matrix(ellipse_curve: np.array, circle_center: (float, float), circle_radius: float,
                  canva: np.array) -> np.array:
    """

    :param ellipse_curve:
    :param circle_center:
    :param circle_radius:
    :param canva: (H, W, 2)
    :return:
    """

    def give_x_to_get_y(circle_center: (float, float), circle_radius: float, x: float):
        """
        according np.root(radius^2 - (x - center[0])^2) + circle_center[1]
        :param circle_center:
        :param circle_radius:
        :param x:
        """
        return np.sqrt(circle_radius ** 2 - (x - circle_center[0]) ** 2 + circle_center[1])

    for index in ellipse_curve:
        circle_y = give_x_to_get_y(circle_center=circle_center, circle_radius=circle_radius, x=index)
        index_x = np.round(index)
        index_y = np.round(ellipse_curve[index])
        canva[int(index_x)][int(index_y)][0] = index_x
        canva[int(index_x)][int(index_y)][1] = circle_y

    return canva
