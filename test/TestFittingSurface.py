import unittest
import numpy as np
from src.fitting_surface import FittingSurface, get_the_point_set_in_the_ellipse
from utils.read_data import read_data
import matplotlib.pyplot as plt
import cv2


class TestFittingSurface(unittest.TestCase):
    def test_fun_get_last_n_point(self):
        image_data = np.arange(0, 100, 1).reshape(10, 10)
        fitting_process = FittingSurface(image_data=image_data)
        lowest_point = fitting_process.obtain_last_n_value_point(number=2, threshold=2)
        self.assertEqual(lowest_point.tolist(), [[0, 2], [0, 3]])

    def test_fun_get_last_n_point_image(self):
        file_name = '../data/pos_2.bin'
        image_data = read_data(file_name=file_name, width=512, height=512, read_type='double')
        fitting_process = FittingSurface(image_data=image_data)
        fitting_process.visualize_point_original_data()
        number_list = [6000, 12000, 24000]

        get_l2_distance = lambda point_location, center_location: (point_location[0] - center_location[0]) ** 2 + (
                point_location[1] - center_location[1]) ** 2

        for index in range(len(number_list)):
            lowest_point = fitting_process.obtain_last_n_value_point(number=number_list[index], threshold=-2300)

            canva = np.zeros(shape=(512, 512))
            pretend_center = (258, 234)
            filtered_by_distant_point = []
            for point in lowest_point:
                distance_from_the_center = get_l2_distance(point_location=point,
                                                           center_location=pretend_center)
                filtered_by_distant_point.append([point[0], point[1], distance_from_the_center])
                canva[point[0], point[1]] = 200

            sorted_filtered_points = sorted(filtered_by_distant_point, key=lambda x: x[2])
            canva_for_second_filter = np.zeros(shape=(512, 512))

            point_for_ellipse_fitting = []
            for location_x, location_y, location_z in sorted_filtered_points[:5000]:
                point_for_ellipse_fitting.append([location_x, location_y])
                canva_for_second_filter[location_x, location_y] = 200

            center, axes, angle = cv2.fitEllipse(np.array(point_for_ellipse_fitting))
            canva = cv2.ellipse(canva, (int(center[1]), int(center[0])), (int(axes[0] // 2), int(axes[1] // 2)),
                                angle=angle,
                                startAngle=0,
                                endAngle=360, thickness=2, color=100)
            plt.subplot(2, len(number_list), index + 1)
            plt.text(0, 650, f"# point: {number_list[index]}")
            plt.imshow(canva)
            plt.subplot(2, len(number_list), index + 1 + len(number_list))
            plt.text(0, 650, f"# point: {number_list[index] // 2}")
            plt.imshow(canva_for_second_filter)

        plt.show()

    def test_show_the_shape_of_surface(self):
        file_name = '../data/pos_2.bin'
        image_data = read_data(file_name=file_name, width=512, height=512, read_type='double')
        fitting_process = FittingSurface(image_data=image_data)
        fitting_process.visualize_point_original_data()

    def test_get_the_set_of_candidate_point(self):
        canva = np.zeros(shape=(512, 512))
        canva = cv2.ellipse(img=canva, center=(240, 236), axes=(362 // 2, 369 // 2), angle=109.59, startAngle=0,
                            endAngle=360, thickness=1, color=100)

        candidate_point = get_the_point_set_in_the_ellipse(ellipse_center=(236, 240), ellipse_axes=(369 // 2, 362 // 2),
                                                           ellipse_angle=109.59, original_map=(512, 512))
        for index in candidate_point:
            canva[index[0], index[1]] = 200
        plt.imshow(canva)
        plt.show()

    def test_split_dataset(self):
        file_name = '../data/pos_2.bin'
        image_data = read_data(file_name=file_name, width=512, height=512, read_type='double')
        fitting_process = FittingSurface(image_data=image_data)

        candidate_point = get_the_point_set_in_the_ellipse(ellipse_center=(236, 240), ellipse_axes=(369 // 2, 362 // 2),
                                                           ellipse_angle=109.59, original_map=(512, 512))
        training_set, validation_set = fitting_process.split_to_train_and_validation_set(candidate_set=candidate_point,
                                                                                         proportion=(0.8, 0.2))

    def test_model(self):
        from src.fitting_surface import Polynomial
        model = Polynomial()
        example = [1, 2]
        output = model(example)
        print(output)

    def test_dataloader(self):
        from torch.utils.data import DataLoader
        from src.fitting_surface import Polynomial
        file_name = '../data/pos_2.bin'
        image_data = read_data(file_name=file_name, width=512, height=512, read_type='double')
        fitting_process = FittingSurface(image_data=image_data)

        candidate_point = get_the_point_set_in_the_ellipse(ellipse_center=(236, 240), ellipse_axes=(369 // 2, 362 // 2),
                                                           ellipse_angle=109.59, original_map=(512, 512))
        training_set, validation_set = fitting_process.split_to_train_and_validation_set(candidate_set=candidate_point,
                                                                                         proportion=(0.8, 0.2))
        train_dataloader = DataLoader(dataset=training_set, batch_size=3, shuffle=True)
        model = Polynomial()
        for location, target in train_dataloader:
            prediction = model(location)
            print(prediction)
            print(target)
            break


    if __name__ == '__main__':
        unittest.main()
