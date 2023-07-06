import unittest
import cv2

from utils.read_data import read_data
from utils.transfer_coordinate_system import cartesian_to_polar
import matplotlib.pyplot as plt
from utils.get_middle import get_middle

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
        plt.imshow(polar_image)
        plt.show()

    def test_get_middle_point(self):
        data_file = '../data/ttestsrc.bin'
        image_data = read_data(file_name=data_file, width=512, height=512, read_type='double')
        middle_x, middle_y = get_middle(img_data=image_data)
        self.assertEqual((middle_x, middle_y), (258, 260))


if __name__ == '__main__':
    unittest.main()
