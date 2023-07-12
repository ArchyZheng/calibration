# %%
import numpy as np
import cv2
from utils.read_data import read_data
import matplotlib.pyplot as plt


class FittingSurface:
    def __init__(self, file_name: str, width: int, height: int, data_type: str):
        """

        :param file_name:
        :param width: bigger than zero
        :param height: bigger than zero
        :param data_type: ['double': 64, 'float': 16, 'int': 8] the type of data the number of bits of each data type
        """
        self.width = width
        self.height = height
        self.data = read_data(file_name=file_name, width=width, height=height, read_type=data_type)

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


if __name__ == "__main__":
    fitting = FittingSurface(file_name='data/pos.bin', width=512, height=512, data_type='double')
    fitting.visualize_point_original_data()
