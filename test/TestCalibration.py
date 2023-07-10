import unittest
from utils.get_the_index_of_point_Threshold import get_the_index_of_point_threshold
from utils.calibration import ellipse_curve, sparse_matrix
import numpy as np
import matplotlib.pyplot as plt
import cv2


class TestCalibration(unittest.TestCase):
    def test_the_location_of_each_row(self):
        image_data = np.zeros(shape=(512, 512))
        image_data[10, :] = 200
        image_data[20, :] = 200
        image_data[30, :] = 200
        image_data[40, :] = 200
        image_data = image_data.T

        row_image = 1
        col_image = 2
        plt.figure()
        plt.subplot(row_image, col_image, 1)
        plt.imshow(image_data)
        candidate_location = get_the_index_of_point_threshold(image_data=image_data, region=[0, 512], threshold=100)
        print(candidate_location)
        plt.show()

    def test_the_location_of_each_row_image(self):
        canva = np.zeros(shape=(512, 512, 3))

        ellipse = cv2.ellipse(img=canva, center=(258, 261), axes=(410 // 2, 354 // 2), angle=89, startAngle=0,
                          endAngle=360, color=(121, 233, 123), thickness=1)
        ellipse_and_circle = cv2.circle(img=ellipse, center=(258, 261), radius=410 // 2, color=(255, 255, 255),
                                thickness=1)
        plt.imshow(ellipse_and_circle)
        plt.show()

    def test_the_image_of_ellipse(self):
        center = (258.4 * 10, 261.00 * 10)
        axes = (410.6/2 * 10, 354.6/2 * 10)
        angle = 89.35

        ellipse_curve_output = ellipse_curve(center=center, axes=axes, angle=angle, shape=(5120, 5120), precision=0.1)
        canva_trial = np.zeros(shape=(5120, 5120))
        for index in ellipse_curve_output:
            canva_trial[int(index)][int(ellipse_curve_output[index])] = 200
        plt.imshow(canva_trial)
        plt.show()



        canva = np.zeros(shape=(5120, 5120, 2))
        output = sparse_matrix(ellipse_curve=ellipse_curve_output, circle_center=(258.4, 261.0), circle_radius=410.6 / 2, canva=canva)
        print(output.shape)



if __name__ == '__main__':
    unittest.main()
