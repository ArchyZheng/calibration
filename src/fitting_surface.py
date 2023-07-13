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

    def obtain_last_n_value_point(self, number: int):
        """
        this function will obtain the lowest n point to get the region for fitting.
        :param number:
        """
        sorted_data = np.sort(self.data.reshape(-1))
        output = []
        while len(output) < number:
            location_point_rows, location_point_cols = np.where(self.data == sorted_data[len(output)])
            point_rows_reshape = location_point_rows.reshape(-1, 1)
            point_cols_reshape = location_point_cols.reshape(-1, 1)
            location_point_list = np.concatenate((point_rows_reshape, point_cols_reshape), 1)
            for point_index in location_point_list:
                output.append(point_index)
        return np.array(output)
