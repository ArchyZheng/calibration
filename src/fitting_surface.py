# %%
import numpy as np
import cv2
from utils.read_data import read_data
import matplotlib.pyplot as plt


class FittingSurface:
    def __init__(self, image_data: str):
        """

        :param image_data:
        """
        self.data = image_data

    def visualize_point_original_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z = self.data
        rows, cols = self.data.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        ax.scatter(x, y, z, s=1, c=z, cmap='jet')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def obtain_last_n_value_point(self, threshold: int, number: int):
        """
        this function will obtain the lowest n point to get the region for fitting.
        :param threshold: the chosen point should bigger or equal to threshold
        :param number:
        """
        sorted_data = np.sort(self.data.reshape(-1))

        filtered_date_index = sorted_data >= threshold
        filtered_data = sorted_data[filtered_date_index]

        output = []
        while len(output) < number:
            location_point_rows, location_point_cols = np.where(self.data == filtered_data[len(output)])
            point_rows_reshape = location_point_rows.reshape(-1, 1)
            point_cols_reshape = location_point_cols.reshape(-1, 1)
            location_point_list = np.concatenate((point_rows_reshape, point_cols_reshape), 1)
            for point_index in location_point_list:
                output.append(point_index)
        return np.array(output)


def get_the_point_set_in_the_ellipse(ellipse_center: (float, float), ellipse_axes: (float, float),
                                     ellipse_angle: float, original_map: (int, int)) -> np.array:
    """
    return the candidate of point location which is inside the ellipse.
    :param ellipse_center:
    :param ellipse_axes:
    :param ellipse_angle:
    :param original_map:
    """
