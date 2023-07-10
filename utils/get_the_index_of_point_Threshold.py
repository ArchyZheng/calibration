# %%
import numpy as np


def get_the_index_of_point_threshold(image_data: np.array, region: [int, int], threshold: float) -> np.array:
    """
    This function will generate the array which will involve the data alongside the first dimension (y-axis).

    :param image_data:
    :param region: demonstrate the top and bottom line of region, the second element is not involved into the region
    :param threshold:
    """

    output = []
    for row_index in range(region[0], region[1], 1):
        row_output = []
        for col_index in range(len(image_data[0])):
            if image_data[row_index, col_index] >= threshold:
                row_output.append([row_index, col_index])
        output.append(row_output)
    return output
