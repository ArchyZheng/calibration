# %%
import numpy as np
import cv2
from utils.read_data import read_data
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split


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

    def split_to_train_and_validation_set(self, candidate_set: np.array, proportion: (float, float)):
        """
        Split the data points into training set and validation set according proportion
        :param candidate_set:
        :param proportion: training, validation
        :returns `train_dataset` and `validation_dataset`
        """
        dataset = Dataset(point_location_list=candidate_set, point=self.data)
        length_training_set = len(dataset) * proportion[0]
        train_dataset, validation_dataset = random_split(dataset,
                                                         [length_training_set, len(dataset) - length_training_set])
        return train_dataset, validation_dataset


class Dataset(Dataset):
    def __init__(self, point_location_list: np.array, point: np.array):
        super().__init__()
        self.point_location_list = point_location_list
        self.point = point

    def __getitem__(self, item):
        index_x = self.point_location_list[item][0]
        index_y = self.point_location_list[item][1]
        return self.point[index_x, index_y]

    def __len__(self):
        return len(self.point_location_list)


def get_the_point_set_in_the_ellipse(ellipse_center: (float, float), ellipse_axes: (float, float),
                                     ellipse_angle: float, original_map: (int, int)) -> np.array:
    """
    return the candidate of point location which is inside the ellipse.
    :param ellipse_center:
    :param ellipse_axes:
    :param ellipse_angle:
    :param original_map:
    """
    radians = np.radians(ellipse_angle)
    rows, cols = original_map
    mesh_map = np.meshgrid(range(rows), range(cols))
    center_x, center_y = ellipse_center
    axes_x, axes_y = ellipse_axes

    inside_point = []

    def ellipse_equation(x, y):
        new_x = (x - center_x) * np.cos(radians) + (y - center_y) * np.sin(radians)
        new_y = (x - center_x) * np.sin(radians) - (y - center_y) * np.cos(radians)
        return (new_x ** 2) / (axes_x ** 2) + (new_y ** 2) / (axes_y ** 2)

    mesh_points = []
    for x in range(rows):
        for y in range(cols):
            mesh_points.append([x, y])

    for x, y in mesh_points:
        value_after_ellipse_equation = ellipse_equation(x, y)
        if value_after_ellipse_equation <= 1:
            inside_point.append([x, y])

    return np.array(inside_point)
