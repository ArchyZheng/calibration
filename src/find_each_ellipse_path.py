import cv2

from utils.read_data import read_data
import matplotlib.pyplot as plt
from utils.dijkstra import normalize_the_image, shortest_path
from utils.transfer_coordinate_system import cartesian_to_polar, polar_to_cartesian, get_location_of_cartesian
from utils.transfer_coordinate_system import get_high_value_point
import numpy as np


def show_the_path_for_each_curve():
    data_file = '../data/imgxy.bin'
    image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
    normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])

    # Because in the process of finding curve, the selected area cannot be well separated from the target curve, so the picture needs to be preliminarily corrected.
    coefficient = 1. / (354.6 / 2) * (410.6 / 2)
    resize_image = cv2.resize(normalized_image, (512, int(512 * coefficient)), cv2.INTER_NEAREST)

    coarse_height, coarse_width = resize_image.shape
    plt.imshow(resize_image)
    plt.show()

    polar_image = cartesian_to_polar(image_tensor=resize_image, original_point=[233, 296], max_radius=240)
    plt.imshow(polar_image)
    plt.show()

    polar_image[:, :10] = 0
    polar_image[140:160, :] = 0
    polar_image[430:450, :] = 0

    begin_point_list = np.array([36, 72, 109, 145, 182, 218, 254, 292, 329, 365, 401, 438, 475])

    # 当前图: 矫正后
    the_shortest_path_list = []
    for index_begin_point in range(len(begin_point_list)):
        begin_point = int(begin_point_list[index_begin_point])
        region_y = (-10, 10)
        sub_image = polar_image[:, begin_point + region_y[0]: begin_point + region_y[1]]
        path, seen = shortest_path(img_data=sub_image, begin=(2, 10), end=(511, 10))
        the_shortest_path_list.append(np.array(path) + [0, begin_point - 10])
    canva = np.zeros(shape=(512, 512))
    for path in the_shortest_path_list:
        for x, y in path:
            canva[x, y] = 200

    plt.imshow(canva)
    plt.show()

    # 将当前坐标矫正回原图 -> 直角坐标系, 缩放
    center_list = []
    axes_list = []
    angle_list = []
    output_list = []
    curve_list = []
    for path in the_shortest_path_list:
        curve = []
        for index in path:
            output_x, output_y = get_location_of_cartesian(polar_theta=index[0], polar_radius=index[1] * 240 / coarse_width,
                                                           polar_center=(298, 260), width=512)
            output_list.append([output_y,
                                output_x * 512 / coarse_height])  # 592 come from coarse adjustment making picture from ellipse to circle.
            curve.append([output_y, output_x * 512 / 592])
        curve_list.append(curve)
        curve = np.array(curve, dtype=int)
        center, axes, angle = cv2.fitEllipse(curve)
        center_list.append(center)
        axes = np.array(axes)
        axes_list.append(axes)
        angle_list.append(angle)

    canva = np.zeros(shape=(512, 512))
    for index in range(len(center_list)):
        center = center_list[index]
        center = np.array(center, dtype=int)
        angle = angle_list[index]
        axes = axes_list[index]
        axes = np.array(axes / 2, dtype=int)
        cv2.ellipse(img=canva, center=[center[1], center[0]], axes=[axes[1], axes[0]], angle=angle, thickness=1,
                    startAngle=0, endAngle=360,
                    color=100)

    # get the dx and dy
    canva[260, 258] = 200

    def get_resolution(interval, save_list: list):
        for outer_index in range(len(interval) - 1):
            outer = interval[outer_index]
            inner_index = outer_index + 1
            inner = interval[inner_index]
            save_list.append(1 / (inner - outer))

    interval_horizontal_list = np.where(canva[260, :] >= 100)
    horizontal_resolution = []
    get_resolution(interval_horizontal_list[0], horizontal_resolution)
    interval_vertical_list = np.where(canva[:, 258] >= 100)
    vertical_resolution = []
    get_resolution(interval_vertical_list[0], vertical_resolution)

    vertical_resolution = np.array(vertical_resolution)
    horizontal_resolution = np.array(horizontal_resolution)

    vertical_resolution.tofile('vertical_resolution.bin')
    horizontal_resolution.tofile('horizontal_resolution.bin')
    np.savetxt('vertical_resolution.csv', vertical_resolution, delimiter=',')
    np.savetxt('horizontal_resolution.csv', horizontal_resolution, delimiter=',')


if __name__ == "__main__":
    show_the_path_for_each_curve()
