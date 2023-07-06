import unittest
import cv2
import numpy as np
from utils.read_data import read_data
from utils.transfer_coordinate_system import cartesian_to_polar
import matplotlib.pyplot as plt
from utils.get_middle import get_middle
from utils.dijkstra import shortest_path
from utils.dijkstra import normalize_the_image


class UtilsFunctionTest(unittest.TestCase):
    def test_shape_of_data(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        self.assertEqual(image_data.shape, (512, 512))

    def test_image_of_data(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        plt.imshow(image_data)
        plt.show()

    def test_cartesian_to_polar_Circle(self):
        data_file_name = '../data/img_1.png'
        image_data = cv2.imread(data_file_name, cv2.IMREAD_GRAYSCALE)
        print(image_data.shape)
        plt.imshow(image_data)
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=(53, 53), max_radius=53)
        plt.imshow(polar_image)
        plt.show()

    def test_cartesian_to_polar_calibration(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        x, y = get_middle(img_data=polar_image)
        plt.imshow(polar_image)
        plt.show()

    def test_get_middle_point(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        middle_x, middle_y = get_middle(img_data=image_data)
        self.assertEqual((middle_x, middle_y), (258, 260))

    def test_find_the_shortest_path(self):
        dummy_map = np.zeros(shape=(10, 10), dtype='double')
        dummy_map[:, 5] = 100  # created for test, the sixth colum in this matrix is equal to 20.
        dummy_map[5, 5] = 0
        dummy_map[5, 6] = 100
        dummy_map[4, 6] = 100
        dummy_map[7, 6] = 100
        the_shortest_path, seen = shortest_path(img_data=dummy_map, begin=(0, 5), end=(9, 5))
        print(the_shortest_path)

    def test_find_the_shortest_path_matrix(self):
        dummy_map = np.zeros(shape=(10, 10), dtype='double')
        dummy_map[:, 5] = 0.01  # created for test, the sixth colum in this matrix is equal to 20.
        dummy_map[5, 5] = 0
        dummy_map[5, 6] = 0.01
        dummy_map[4, 6] = 0.01
        dummy_map[6, 6] = 0.01
        the_shortest_path, seen = shortest_path(img_data=dummy_map, begin=(0, 5), end=(9, 5))

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(dummy_map)
        cost_matrix = np.zeros_like(dummy_map)
        for index in seen:
            cost_matrix[index] = seen[index]
        plt.subplot(1, 3, 2)
        plt.imshow(cost_matrix)
        path_matrix = np.zeros_like(dummy_map)
        for index in the_shortest_path:
            path_matrix[index] = 200
        plt.subplot(1, 3, 3)
        plt.imshow(path_matrix)
        plt.show()

    def test_find_the_shortest_path_figure(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        plt.imshow(polar_image)
        plt.show()
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(80, 2), end=(31, 509))
        matrix = np.zeros_like(polar_image)
        for index in seen:
            matrix[index] = seen[index]

        plt.imshow(matrix)
        plt.show()

    def test_normalize_the_image(self):
        matrix = np.arange(0, 100, 1, dtype=float)
        matrix = matrix.reshape(10, 10)
        print(matrix)
        normalized_image = normalize_the_image(image_data=matrix, threshold=[0, 0.9])
        print(normalized_image)

        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])
        plt.imshow(normalized_image)
        plt.show()

    def test_the_cost_matrix_after_normalize_the_image(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])
        polar_image = cartesian_to_polar(image_tensor=normalized_image, original_point=[258, 260], max_radius=240)
        polar_image[:, :10] = 0
        polar_image[252:258, :] = 0
        plt.imshow(polar_image)
        plt.show()
        # the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 33), end=(509, 32))
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 381), end=(511, 380))

        path_matrix = np.zeros_like(polar_image)
        for index in the_shortest_path:
            path_matrix[index] = 200

        plt.imshow(path_matrix)
        plt.show()


if __name__ == '__main__':
    unittest.main()
