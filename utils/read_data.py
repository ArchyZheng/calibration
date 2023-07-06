# %%
import numpy as np


def read_data(file_name: str, width: int, height: int, read_type: str) -> np.array:
    """
    read file created from matlab, and return the numpy array.

    :param read_type: using different type to read the data, according the width of one point
    :param file_name:
    :param width: must bigger than zero
    :param height: must begger than zero
    """
    image_tensor: np.array = np.fromfile(file_name, dtype=read_type)
    return image_tensor.reshape(width, height).T
