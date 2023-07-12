import cv2

from utils.read_data import read_data
import matplotlib.pyplot as plt
from utils.dijkstra import normalize_the_image, shortest_path
from utils.transfer_coordinate_system import cartesian_to_polar, polar_to_cartesian, get_location_of_cartesian, \
    get_y_on_the_circle
from utils.transfer_coordinate_system import get_high_value_point
import numpy as np


def show_the_path_for_each_curve():
    data_file = '../data/ttestsrc.bin'
    image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
    normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])

    # Because in the process of finding curve, the selected area cannot be well separated from the target curve, so the picture needs to be preliminarily corrected.
    coefficient = 1. / (354.6 / 2) * (410.6 / 2)
    resize_image = cv2.resize(normalized_image, (int(512 * coefficient), 512), cv2.INTER_NEAREST)

    polar_image = cartesian_to_polar(image_tensor=resize_image, original_point=[298, 260], max_radius=240)
    polar_image[:, :10] = 0
    polar_image[254:262, :] = 200

    begin_point_list = np.array([45, 88, 128, 170, 212, 254, 298, 339, 381, 424, 467, 509, 550])

    # 当前图: 矫正后
    the_shortest_path_list = []
    for index_begin_point in range(len(begin_point_list)):
        begin_point = int(begin_point_list[index_begin_point])
        region_y = (-10, 10)
        sub_image = polar_image[:, begin_point + region_y[0]: begin_point + region_y[1]]
        path, seen = shortest_path(img_data=sub_image, begin=(2, 10), end=(511, 10))
        the_shortest_path_list.append(np.array(path) + [0, begin_point - 10])

    # 将当前坐标矫正回原图 -> 直角坐标系, 缩放
    center_list = []
    axes_list = []
    angle_list = []
    output_list = []
    curve_list = []
    for path in the_shortest_path_list:
        curve = []
        for index in path:
            output_x, output_y = get_location_of_cartesian(polar_theta=index[0], polar_radius=index[1] * 240 / 592,
                                                           polar_center=(298, 260), width=512)
            output_list.append([output_y, output_x * 512 / 592])
            curve.append([output_y, output_x * 512 / 592])
        curve_list.append(curve)
        curve = np.array(curve, dtype=int)
        center, axes, angle = cv2.fitEllipse(curve)
        center_list.append(center)
        axes = np.array(axes)
        axes_list.append(axes)
        angle_list.append(angle)

    canva = np.zeros(shape=(512, 512))
    curve_to_circle = []
    for index in range(len(center_list)):
        center = center_list[index]
        center = np.array(center, dtype=int)
        angle = angle_list[index]
        axes = axes_list[index]
        axes = np.array(axes / 2, dtype=int)
        edge_curve = curve_list[index]
        for edge_y, edge_x in edge_curve:
            circle_y = get_y_on_the_circle(x=edge_x, center=center, radius=axes[1])
            if circle_y[0] is not None:
                curve_to_circle.append([edge_x, edge_y, circle_y])
    canva = np.zeros(shape=(512 + 100, 512 + 100))
    for edge_x, edge_y, circle_y in curve_to_circle:
        print(f"edge: {edge_x}, {edge_y}, circle_y: {circle_y}")
        canva[int(edge_x), np.array(circle_y, dtype=int) + 100] = 200
    plt.imshow(canva)
    plt.show()


if __name__ == "__main__":
    show_the_path_for_each_curve()
