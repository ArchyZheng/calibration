# %% This file is created for dataloader module.
import cv2

from utils.read_data import read_data
from utils.transfer_coordinate_system import cartesian_to_polar, polar_to_cartesian
import numpy as np


class Dataloader:
    def __init__(self):
        self.original_image = None  # original data is always in the cartesian coordinate system.
        self.polar_image = None
        self.resized_image = None

    def read_image(self, file_path: str, width: int, height: int, read_type: str):
        original_image = read_data(file_name=file_path, height=height, width=width, read_type=read_type)
        original_image_normalized = (original_image - np.min(original_image)) / (
                np.max(original_image) - np.min(original_image))

        self.original_image = original_image_normalized

        return self.original_image

    def transform_car_to_polar(self, center: [int, int], radius: int, src: np.array):
        """
        transform the coordinate system from cartesian to polar
        :param src:
        :param center:
        :param radius:
        """
        self.polar_image = cartesian_to_polar(image_tensor=src, original_point=center,
                                              max_radius=radius)

    def resize_image(self, new_size: [int, int], original_image: np.array):
        self.resized_image = cv2.resize(src=original_image, dsize=new_size, interpolation=cv2.INTER_NEAREST)
