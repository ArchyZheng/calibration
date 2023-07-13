import unittest
import numpy as np
from src.fitting_surface import FittingSurface


class TestFittingSurface(unittest.TestCase):
    def test_fun_get_last_n_point(self):
        image_data = np.arange(0, 100, 1).reshape(10, 10)
        fitting_process = FittingSurface(image_data=image_data)
        lowest_point = fitting_process.obtain_last_n_value_point(number=2)
        self.assertEqual(lowest_point.tolist(), [[0, 0], [0, 1]])


if __name__ == '__main__':
    unittest.main()
