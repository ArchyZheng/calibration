import unittest
import numpy as np
from src.fitting_surface import FittingSurface
from utils.read_data import read_data
import matplotlib.pyplot as plt
import cv2


class TestFittingSurface(unittest.TestCase):
    def test_fun_get_last_n_point(self):
        image_data = np.arange(0, 100, 1).reshape(10, 10)
        fitting_process = FittingSurface(image_data=image_data)
        lowest_point = fitting_process.obtain_last_n_value_point(number=2)
        self.assertEqual(lowest_point.tolist(), [[0, 0], [0, 1]])

    def test_fun_get_last_n_point_image(self):
        file_name = '../data/pos.bin'
        image_data = read_data(file_name=file_name, width=512, height=512, read_type='double')
        fitting_process = FittingSurface(image_data=image_data)
        # fitting_process.visualize_point_original_data()
        number_list = [6000]

        for index in range(len(number_list)):
            lowest_point = fitting_process.obtain_last_n_value_point(number=number_list[index])

            canva = np.zeros(shape=(512, 512))
            for point in lowest_point:
                canva[point[0], point[1]] = 200

            center, axes, angle = cv2.fitEllipse(lowest_point)
            canva = cv2.ellipse(canva, (int(center[1]), int(center[0])), (int(axes[0] // 2), int(axes[1] // 2)), angle=angle,
                                startAngle=0,
                                endAngle=360, thickness=2, color=100)
            plt.subplot(1, len(number_list), index + 1)
            plt.imshow(canva)

        plt.show()


if __name__ == '__main__':
    unittest.main()
