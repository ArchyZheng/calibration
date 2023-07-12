import unittest

import matplotlib.pyplot as plt

from utils.transfer_coordinate_system import get_y_on_the_circle, draw_a_circle
import numpy as np
import cv2


class TestCalculation(unittest.TestCase):
    def test_y_coordinate_given_x(self):
        center = (5, 5)
        radius = 5
        y = get_y_on_the_circle(x=5, radius=radius, center=center)
        self.assertEqual(y, [10, 0])

    def test_y_coordinate_on_the_graph(self):
        canva = np.zeros(shape=(512, 512))
        center = (258, 260)
        cv2.circle(img=canva, center=center, radius=40, color=200,
                   thickness=2)  # This error comes from the draw the circle with the cv2.
        x = 258
        y = get_y_on_the_circle(x=x, center=(258, 260), radius=40)
        canva[np.array(y, dtype=int), x] = 100  # this is right
        canva[center[0], center[1]] = 100

        plt.imshow(canva)
        plt.show()
        # we need careful the definition of (x, y)

    def test_draw_a_circle(self):
        canva = np.zeros(shape=(512, 512))
        center = (258, 260)
        draw_a_circle(image=canva, center=center, radius=40)
        canva[center[0], center[1]] = 100
        x = 258
        y = get_y_on_the_circle(x=x, center=(258, 260), radius=40)
        canva[x, np.array(y, dtype=int)] = 100  # this is right
        plt.imshow(canva)
        plt.show()




if __name__ == '__main__':
    unittest.main()
