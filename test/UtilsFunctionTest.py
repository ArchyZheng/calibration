import unittest
import cv2
import numpy as np
from utils.read_data import read_data
from utils.transfer_coordinate_system import cartesian_to_polar
import matplotlib.pyplot as plt
from utils.get_middle import get_middle
from utils.dijkstra import shortest_path
from utils.dijkstra import normalize_the_image
from utils.dijkstra import get_center_of_mass
from utils.filtering import generate_gaussian_kernel
from utils.filtering import convolve2d
from utils.filtering import convolve1d
from utils.filtering import generate_one_dimension_kernel
from utils.transfer_coordinate_system import polar_to_cartesian
from utils.transfer_coordinate_system import polar_vector_to_cartesian_vector
from utils.transfer_coordinate_system import get_high_value_point
from utils.fitting import fitting_ellipse


class UtilsFunctionTest(unittest.TestCase):
    def test_shape_of_data(self):
        data_file = '../data/imgxy.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        self.assertEqual(image_data.shape, (512, 512))

    def test_data_of_image(self):
        # data_file = '../data/ttestsrc.bin'
        data_file = '../data/pos_pixel.bin'
        pos_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        xy_file = '../data/ttestsrc.bin'
        xy_data = read_data(file_name=xy_file, width=512, height=512, read_type='double')
        convert_xy_data = xy_data[512::-1, :]
        plt.subplot(1, 4, 1)
        plt.imshow(convert_xy_data + pos_data)
        plt.subplot(1, 4, 2)
        plt.imshow(convert_xy_data + 2000)
        plt.subplot(1, 4, 3)
        plt.imshow(xy_data)
        plt.subplot(1, 4, 4)
        plt.imshow(pos_data)
        plt.show()

    def test_cartesian_to_polar_circle_image(self):
        data_file_name = '../data/img_1.png'
        image_data = cv2.imread(data_file_name, cv2.IMREAD_GRAYSCALE)
        print(image_data.shape)
        plt.imshow(image_data)
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=(53, 53), max_radius=53)
        plt.imshow(polar_image)
        plt.show()

    def test_cartesian_to_polar_calibration_image(self):
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

    def test_find_the_shortest_path_Figure(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        plt.imshow(polar_image)
        plt.show()
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 32), end=(511, 32))
        path_matrix = np.zeros_like(polar_image)
        for index in the_shortest_path:
            print(index)
            path_matrix[index] = 200

        plt.imshow(path_matrix)
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
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(image_data)
        normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])
        polar_image = cartesian_to_polar(image_tensor=normalized_image, original_point=[258, 260], max_radius=240)
        # polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        polar_image[:, :10] = 0
        polar_image[254:262, :] = 200
        plt.subplot(1, 3, 2)
        plt.imshow(polar_image)
        # the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 33), end=(509, 32))
        # the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 381), end=(511, 380))
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 32), end=(511, 32))

        path_matrix = np.zeros_like(polar_image)
        for index in the_shortest_path:
            polar_image[index] = 255.

        plt.subplot(1, 3, 3)
        plt.imshow(polar_image)
        plt.show()

    def test_center_of_mass_Vector(self):
        vector = np.array([1, 2, 3, 4, 5])
        index = np.array([0, 1, 2, 3, 4])
        mass_index = get_center_of_mass(vector=vector, index_list=index)
        self.assertEqual(mass_index, 2)

    def test_center_of_mass_figure(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])
        polar_image = cartesian_to_polar(image_tensor=normalized_image, original_point=[258, 260], max_radius=240)

        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 381), end=(511, 380))

        new_path = []
        for x, y in the_shortest_path:
            vector = polar_image[x, y - 10:y + 10]  # the 10 means the half width of the scope
            index = np.arange(y - 10, y + 10, 1)
            mass_index = get_center_of_mass(vector=vector, index_list=index)
            new_path.append([x, mass_index])

    def test_zoom_a_figure(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])
        polar_image = cartesian_to_polar(image_tensor=normalized_image, original_point=[258, 260], max_radius=240)
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 381), end=(511, 380))

        new_path = []
        for x, y in the_shortest_path:
            vector = polar_image[x, y - 3:y + 3]  # the 10 means the half width of the scope
            index = np.arange(y - 3, y + 3, 1)
            mass_index = get_center_of_mass(vector=vector, index_list=index)
            new_path.append([x, mass_index])

        polar_image = cv2.resize(polar_image, (5120, 512), interpolation=cv2.INTER_NEAREST)
        plt.imshow(polar_image)
        plt.show()

        path_image = np.zeros_like(polar_image)
        for x, y in new_path:
            path_image[x, int(10 * y)] = 255
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(path_image)
        plt.subplot(1, 2, 2)
        plt.imshow(polar_image)
        plt.show()

    def test_the_kernel_of_gaussian(self):
        kernel = generate_gaussian_kernel(l=3, sig=1.)
        plt.imshow(kernel)
        plt.show()

    def test_gaussian_filter(self):
        image_file = "../data/ttestsrc.bin"
        image_data = read_data(file_name=image_file, width=512, height=512, read_type='double')
        normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])
        polar_image = cartesian_to_polar(image_tensor=normalized_image, original_point=[258, 260], max_radius=240)

        plt.figure()
        kernel = generate_gaussian_kernel(l=5, sig=1.)
        after_gaussian_filter_image = convolve2d(kernel=kernel, image_data=polar_image)
        plt.subplot(1, 2, 1)
        plt.imshow(polar_image)
        plt.subplot(1, 2, 2)
        plt.imshow(after_gaussian_filter_image)
        plt.show()

    def test_filter2D(self):
        array = np.ones(shape=(100, 100))
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        dst = convolve2d(image_data=array, kernel=kernel)
        print(dst)

    def test_find_path_after_gaussian_filter(self):
        image_file = "../data/ttestsrc.bin"
        image_data = read_data(file_name=image_file, width=512, height=512, read_type='double')
        normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])
        polar_image = cartesian_to_polar(image_tensor=normalized_image, original_point=[258, 260], max_radius=240)

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.imshow(polar_image)
        kernel = generate_gaussian_kernel(l=5, sig=1.)
        after_gaussian_filter_image = convolve2d(kernel=kernel, image_data=polar_image)

        plt.subplot(1, 3, 2)
        plt.imshow(after_gaussian_filter_image)
        the_shortest_path, seen = shortest_path(img_data=after_gaussian_filter_image, begin=(2, 381), end=(511, 380))
        path_matrix = np.zeros_like(after_gaussian_filter_image)
        for index in the_shortest_path:
            path_matrix[index] = 200
        plt.subplot(1, 3, 3)
        plt.imshow(path_matrix)
        plt.show()

    def test_filter(self):
        curve = np.array([1, 2, 3])
        kernel = generate_one_dimension_kernel(l=3, sig=1.)
        new_curve = convolve1d(kernel=kernel, curve=curve)
        print(new_curve)

    def test_generate_one_dimension_kernel(self):
        kernel = generate_one_dimension_kernel(l=3, sig=1.)
        print(kernel)

    def test_smooth_the_curve(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(0, 381), end=(512, 380))

        curve = []
        for x, y in the_shortest_path:
            curve.append(y)

        # filter part:
        kernel = generate_one_dimension_kernel(l=5, sig=2.)
        new_curve = convolve1d(kernel=kernel, curve=curve)
        plt.figure()
        image_matrix = np.zeros_like(polar_image)
        for index in range(len(the_shortest_path)):
            image_matrix[the_shortest_path[index][0], int(new_curve[index])] = 200
        plt.subplot(1, 2, 2)
        plt.imshow(image_matrix)
        plt.subplot(1, 2, 1)
        original_image = np.zeros_like(polar_image)
        for index in the_shortest_path:
            original_image[index[0], index[1]] = 200
        plt.imshow(original_image)
        plt.show()

    def test_polar_system_to_cartesian(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        cartesian_image = polar_to_cartesian(polar_tensor=polar_image, original_point=[258, 260], max_radius=240)
        plt.imshow(cartesian_image)
        plt.show()

    def test_polar_vector_to_cartesian_vector(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 381), end=(511, 380))

        cartesian_points = polar_vector_to_cartesian_vector(polar_vector=the_shortest_path, max_radius=240)
        path_matrix = np.zeros_like(polar_image)
        for x, y in cartesian_points:
            path_matrix[int(x), int(y)] = 200

        plt.imshow(path_matrix)
        plt.show()

    def test_polar_vector_to_cartesian(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 381), end=(511, 380))

        plt.figure()
        path_matrix = np.zeros_like(polar_image)
        for index in the_shortest_path:
            path_matrix[index] = 200
        plt.subplot(1, 3, 1)
        plt.imshow(path_matrix)

        cartesian_image = polar_to_cartesian(polar_tensor=path_matrix, original_point=[258, 260], max_radius=240)
        plt.subplot(1, 3, 2)
        plt.imshow(cartesian_image)
        the_index_of_high = get_high_value_point(image_picture=cartesian_image, threshold=100)
        path_matrix_reformed = np.zeros_like(cartesian_image)
        for index in the_index_of_high:
            path_matrix_reformed[index[0], index[1]] = 200
        plt.subplot(1, 3, 3)
        plt.imshow(path_matrix_reformed)
        plt.show()

    def test_pick_the_high_value_point(self):
        array = np.arange(0, 100, 1)
        array = array.reshape(10, 10)
        the_index_of_high = get_high_value_point(image_picture=array, threshold=49)
        plt.imshow(array)
        plt.show()
        matrix = np.zeros_like(array)
        for index in the_index_of_high:
            matrix[index[0], index[1]] = 10
        plt.imshow(matrix)
        plt.show()

        print(the_index_of_high)

    def test_fitting_ellipse(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        polar_image = cartesian_to_polar(image_tensor=image_data, original_point=[258, 260], max_radius=240)
        the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 381), end=(511, 380))
        # the_shortest_path, seen = shortest_path(img_data=polar_image, begin=(2, 126), end=(511, 126))

        path_matrix = np.zeros_like(polar_image)
        for index in the_shortest_path:
            path_matrix[index] = 200

        cartesian_image = polar_to_cartesian(polar_tensor=path_matrix, original_point=[258, 260], max_radius=240)
        plt.imshow(cartesian_image)
        plt.show()
        np.save('cartesian_image.txt', cartesian_image)
        the_index_of_high = get_high_value_point(image_picture=cartesian_image, threshold=100)
        the_index_of_high = np.array(the_index_of_high)

        parameters_of_ellipse_AMS = fitting_ellipse(the_index_of_high)
        parameters_of_ellipse = cv2.fitEllipse(the_index_of_high)
        parameters_of_ellipse_Direct = cv2.fitEllipseDirect(the_index_of_high)
        print(
            f"parameters_of_ellipse: {parameters_of_ellipse} \n parameters_of_ellipse_Direct: {parameters_of_ellipse_Direct} \n parameters_of_ellipse_AMS: {parameters_of_ellipse_AMS}")

    def test_draw_an_ellipse(self):
        img = np.zeros((512, 512, 3), np.uint8)
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        normalized_image = normalize_the_image(image_data=image_data, threshold=[0.01, 0.999])
        image_data = np.expand_dims(normalized_image, axis=2).repeat(3, axis=2)

        original_image = np.load('cartesian_image.txt.npy')
        color_image = np.expand_dims(original_image, axis=2).repeat(3, axis=2)

        c = np.zeros_like(img)
        img = cv2.ellipse(img=original_image, center=(258, 261), axes=(410 // 2, 354 // 2), angle=89, startAngle=0,
                          endAngle=360, color=(121, 233, 123),
                          thickness=1)
        img_circle = cv2.circle(img=img, center=(258, 261), radius=410 // 2, color=(255, 255, 255),
                                thickness=1)
        plt.imshow(img_circle)
        plt.show()


if __name__ == '__main__':
    unittest.main()
