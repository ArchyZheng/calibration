# %%
import sys
import cv2
from PySide6 import QtCore, QtWidgets, QtGui
from src.Dataloader import Dataloader
import matplotlib.pyplot as plt
from PIL import Image, ImageQt
import numpy as np
from utils.dijkstra import shortest_path
from utils.transfer_coordinate_system import get_location_of_cartesian


def open_image_by_plt(image_data: np.array):
    plt.imshow(image_data)
    plt.show()


def get_resolution(interval, list_save):
    for outer_index in range(len(interval) - 1):
        outer = interval[outer_index]
        inner_index = outer_index + 1
        inner = interval[inner_index]
        list_save.append(0.5 / (inner - outer))


class DataloaderUi(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.horizontal_resolution = None
        self.vertical_resolution = None
        self.ellipse_canva = None
        self.new_center = None
        self.ellipse_curve_list = None
        self.ellipse_axes_list = None
        self.ellipse_center_list = None
        self.ellipse_angle_list = None
        self.shortest_path_list = None
        self.anchor_list = None
        self.file_name = None
        self.dataloader = Dataloader()
        self.center_x = None
        self.center_y = None
        self.image_navigation_list = []

        self.button_select_file = QtWidgets.QPushButton("Click for select file")
        self.select_button = QtWidgets.QCheckBox("mode of finding center of circle")
        self.label_circle_center = QtWidgets.QLabel("The center of lens:")
        self.layout_select_button = QtWidgets.QHBoxLayout()

        self.layout_select_button.addWidget(self.select_button)
        self.layout_select_button.addWidget(self.label_circle_center)
        self.file_selector = QtWidgets.QFileDialog()
        self.image_window = QtWidgets.QLabel()
        self.button_popup_plt_to_selection_center = QtWidgets.QPushButton(
            "Please point out the center!")  # TODO: change the color of push button!
        # <<<< second line <<<<
        self.label_center_x = QtWidgets.QLabel()
        self.label_center_y = QtWidgets.QLabel()
        self.lineEdit_radius = QtWidgets.QLineEdit()
        self.button_transform_coordinate_cartesian_to_polar = QtWidgets.QPushButton(
            "Cartesian to Polar Coordinate System")

        self.layout_center = QtWidgets.QHBoxLayout()
        self.layout_center.addWidget(self.label_center_x)
        self.layout_center.addWidget(self.label_center_y)
        self.layout_center.addWidget(self.lineEdit_radius)
        self.layout_center.addWidget(self.button_transform_coordinate_cartesian_to_polar)

        # >>>> second line >>>>

        # <<<< Slider part <<<<
        self.slider_adjust_width = QtWidgets.QSlider()
        self.slider_adjust_width.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_adjust_width.valueChanged.connect(self.resize_the_image)
        self.label_of_slider_number = QtWidgets.QLabel()
        self.layout_slider = QtWidgets.QHBoxLayout()
        self.layout_slider.addWidget(self.label_of_slider_number)
        self.layout_slider.addWidget(self.slider_adjust_width)
        # >>>> Slider part >>>>

        # >>>> anchor part >>>>
        self.button_set_anchor = QtWidgets.QPushButton("Press to open image shower")
        self.button_find_curve_polar = QtWidgets.QPushButton("Find Curve")
        self.lineedit_anchor = QtWidgets.QLineEdit()
        self.layout_anchor = QtWidgets.QHBoxLayout()
        self.layout_anchor.addWidget(self.button_set_anchor)
        self.layout_anchor.addWidget(self.lineedit_anchor)
        self.layout_anchor.addWidget(self.button_find_curve_polar)
        # <<<< anchor part <<<<

        # >>>> finding ellipse part >>>>
        self.button_show_ellipse = QtWidgets.QPushButton("Show ellipse on the image")
        # <<<< finding ellipse part <<<<

        # <<<<<  Layout setting  <<<<<
        self.layout_main = QtWidgets.QVBoxLayout(self)
        self.layout_main.addWidget(self.image_window)
        self.layout_main.addWidget(self.button_select_file)
        self.layout_main.addLayout(self.layout_select_button)
        # >>>>>  Layout setting  >>>>>
        self.button_select_file.clicked.connect(self.open_file_list)
        self.button_popup_plt_to_selection_center.clicked.connect(self.popup_plt_image)
        self.button_transform_coordinate_cartesian_to_polar.clicked.connect(
            self.transform_coordinate_cartesian_to_polar)
        self.button_set_anchor.clicked.connect(self.open_reside_image_for_anchor)
        self.button_find_curve_polar.clicked.connect(self.find_path)
        self.button_show_ellipse.clicked.connect(self.find_ellipse)

    @QtCore.Slot()
    def open_file_list(self):
        self.file_name = self.file_selector.getOpenFileNames(parent=self)
        try:
            self.dataloader.read_image(file_path=self.file_name[0][0], width=512, height=512, read_type='double')
        except ValueError:
            log_dialog = QtWidgets.QDialog()
            log_dialog.setWindowTitle("The file is not matrix or the shape of matrix is not right!")
            log_dialog.show()
        # show the image on the image_windows
        self.show_image(image_data=self.dataloader.original_image, temp_file_name="from_bin_array.jpg")
        self.layout_main.addWidget(self.button_popup_plt_to_selection_center)

    def show_image(self, image_data: np.array, temp_file_name: str):
        """
        show the image on the image_window
        :param temp_file_name: the name of picture, for image navigation
        :param image_data: image data represented by the numpy array
        """
        intermediate_image_filename = temp_file_name
        self.image_navigation_list.append(temp_file_name)
        plt.imsave(intermediate_image_filename, image_data)
        img = Image.open(intermediate_image_filename)
        img_qt = ImageQt.ImageQt(img)
        img_pixmap = QtGui.QPixmap.fromImage(img_qt)
        self.image_window.setPixmap(img_pixmap)

    @QtCore.Slot()
    def popup_plt_image(self):
        """
        show the image_figure and add the interaction of mouse click to pick the center point location.
        """
        image_data = self.dataloader.original_image
        fig = plt.gcf()
        image_plot = plt.imshow(image_data)
        cid = fig.canvas.mpl_connect('button_press_event', self.point_center)
        self.layout_main.addLayout(self.layout_center)
        plt.show()

    def point_center(self, event):
        """
        this is muse event, it will change center_x and center_y label according mouse location.
        ref: https://stackoverflow.com/questions/15721094/detecting-mouse-event-in-an-image-with-matplotlib
        :param event:
        """
        if event.xdata is not None and event.ydata is not None:
            self.center_x = event.xdata
            self.center_y = event.ydata
        self.label_center_x.setText(f"center x: {self.center_x}")
        self.label_center_y.setText(f"center y: {self.center_y}")

    def transform_coordinate_cartesian_to_polar(self):
        radius = self.lineEdit_radius.text()
        self.dataloader.transform_car_to_polar(center=[int(self.center_x), int(self.center_y)], radius=int(radius),
                                               src=self.dataloader.original_image)
        assert self.dataloader.polar_image is not None
        self.show_image(image_data=self.dataloader.polar_image, temp_file_name="polar_image.jpg")
        basic_width = self.dataloader.polar_image.shape[1]
        width_range = 100
        self.slider_adjust_width.setRange(basic_width - width_range, basic_width + width_range)
        self.slider_adjust_width.setSingleStep(1)
        self.slider_adjust_width.setSliderPosition(basic_width)
        self.label_of_slider_number.setText(f"Width: {self.slider_adjust_width.value()}")
        self.layout_main.addLayout(self.layout_slider)

        self.layout_main.addLayout(self.layout_anchor)

    def resize_the_image(self):
        new_width = self.slider_adjust_width.value()
        self.label_of_slider_number.setText(f"Width: {new_width}")
        new_size = [int(new_width), self.dataloader.polar_image.shape[0]]
        self.dataloader.resize_image(original_image=self.dataloader.original_image, new_size=new_size)
        # >>>> SOBEL START >>>>
        if self.select_button.isChecked():
            self.dataloader.resized_image = cv2.GaussianBlur(self.dataloader.resized_image, (5, 5), 0)
            # self.dataloader.resized_image = cv2.Sobel(self.dataloader.resized_image, -1, 1, 0)
            # self.dataloader.resized_image = np.where(self.dataloader.resized_image > 0, 0,
            #                                          -1 * self.dataloader.resized_image)
            self.dataloader.resized_image = 255 - self.dataloader.resized_image
        # <<<< SOBEL END <<<<
        self.new_center = [int(self.center_x * new_width // self.dataloader.original_image.shape[0]),
                           int(self.center_y)]
        print(self.new_center)
        self.dataloader.transform_car_to_polar(
            center=[self.new_center[0], self.new_center[1]],
            src=self.dataloader.resized_image,
            radius=int(self.lineEdit_radius.text()))
        self.show_image(image_data=self.dataloader.polar_image, temp_file_name="resize_image.jpg")

    def open_reside_image_for_anchor(self):
        fig = plt.gcf()
        open_image_by_plt(self.dataloader.polar_image)
        cid = fig.canvas.mpl_connect('button_press_event', self.set_anchor)
        self.anchor_list = []

    def set_anchor(self, event):
        if event.xdata is not None and event.ydata is not None:
            y = event.xdata
            self.anchor_list.append(y)
            print(self.anchor_list)

    def find_path(self):
        assert self.anchor_list is not None
        self.shortest_path_list = []
        for index_begin_point in range(len(self.anchor_list)):
            begin_point = int(self.anchor_list[index_begin_point])
            region_y = (-10, 10)
            sub_image = self.dataloader.polar_image[:, begin_point + region_y[0]:begin_point + region_y[1]]
            path, seen = shortest_path(img_data=sub_image, begin=(0, 10), end=(511, 10))
            self.shortest_path_list.append(np.array(path) + [0, begin_point - 10])
        canva = np.zeros_like(self.dataloader.polar_image)
        for path in self.shortest_path_list:
            for x, y in path:
                canva[x, y] = 200
        plt.imshow(canva + self.dataloader.polar_image)
        plt.show()
        self.layout_main.addWidget(self.button_show_ellipse)

    def find_ellipse(self):
        self.ellipse_center_list = []
        self.ellipse_axes_list = []
        self.ellipse_angle_list = []
        self.ellipse_curve_list = []
        for path in self.shortest_path_list:
            curve = []
            for index in path:
                radius = int(self.lineEdit_radius.text())
                new_width = int(self.slider_adjust_width.value())
                output_x, output_y = get_location_of_cartesian(polar_theta=index[0],
                                                               polar_radius=index[1] * radius / new_width,
                                                               polar_center=[self.new_center[1], self.new_center[0]],
                                                               width=self.dataloader.polar_image.shape[0])
                output_y_reformed = output_y * self.dataloader.original_image.shape[1] / new_width
                curve.append([output_y_reformed, output_x])
            self.ellipse_curve_list.append(curve)
            curve = np.array(curve, dtype=int)
            center, axes, angle = cv2.fitEllipse(curve)
            self.ellipse_center_list.append(center)
            axes = np.array(axes)
            self.ellipse_axes_list.append(axes)
            self.ellipse_angle_list.append(angle)
        self.ellipse_canva = np.zeros(shape=self.dataloader.original_image.shape)
        for index in range(len(self.ellipse_center_list)):
            center = self.ellipse_center_list[index]
            center = np.array(center, dtype=int)
            angle = self.ellipse_angle_list[index]
            axes = self.ellipse_axes_list[index]
            axes = np.array(axes / 2, dtype=int)
            if self.select_button.isChecked():
                cv2.ellipse(img=self.ellipse_canva, center=[center[0], center[1]], axes=[axes[0], axes[1]], angle=angle,
                            thickness=1,
                            startAngle=0, endAngle=360,
                            color=200)
            else:
                cv2.ellipse(img=self.ellipse_canva, center=[center[0], center[1]], axes=[axes[0], axes[1]], angle=angle,
                            thickness=1,
                            startAngle=0, endAngle=360,
                            color=200)
        self.ellipse_canva[int(self.center_y), int(self.center_x)] = 200
        plt.imshow(self.ellipse_canva + self.dataloader.original_image)
        plt.show()
        if self.select_button.isChecked():
            print(self.ellipse_center_list[0])
            self.label_circle_center.setText(f"The center of lens: {self.ellipse_center_list[0]}")
        else:
            self.get_vertical_horizontal_resolution()

    def get_vertical_horizontal_resolution(self):
        self.vertical_resolution = []
        self.horizontal_resolution = []
        assert self.ellipse_canva is not None
        interval_horizontal_list = np.where(self.ellipse_canva[int(self.center_y), :] >= 100)
        get_resolution(interval=interval_horizontal_list[0], list_save=self.horizontal_resolution)
        interval_vertical_list = np.where(self.ellipse_canva[:, int(self.center_x)] >= 100)
        get_resolution(interval=interval_vertical_list[0], list_save=self.vertical_resolution)

        def mapping(left_most, right_most, mapping, interval_list, resolution_list):
            summation = []
            resolution_index = -1
            for x in range(left_most, right_most + 1):
                if x in interval_list[0][:] and x != interval_list[0][-1]:
                    resolution_index += 1

                summation.append(resolution_list[resolution_index])
                mapping[x] = sum(summation)

        vertical_map = np.zeros(shape=(512,))
        left_most = interval_vertical_list[0][0]
        right_most = interval_vertical_list[0][-1]
        mapping(left_most=left_most, right_most=right_most, mapping=vertical_map, interval_list=interval_vertical_list,
                resolution_list=self.vertical_resolution)

        bias = vertical_map[int(self.center_y)]
        vertical_numpy = np.array(vertical_map)
        vertical_numpy_insert_nan = np.where(vertical_numpy == 0, np.nan, vertical_numpy - bias)

        horizontal_map = np.zeros(shape=(512,))
        top_most = interval_horizontal_list[0][0]
        bottom_most = interval_horizontal_list[0][-1]
        mapping(left_most=top_most, right_most=bottom_most, mapping=horizontal_map,
                interval_list=interval_horizontal_list,
                resolution_list=self.horizontal_resolution)
        bias = horizontal_map[int(self.center_x)]
        horizontal_numpy = np.array(horizontal_map)
        horizontal_numpy_insert_nan = np.where(horizontal_numpy == 0, np.nan, horizontal_numpy - bias)

        np.savetxt('../data/horizontal_mapping_1.csv', horizontal_numpy_insert_nan, delimiter=',')
        np.savetxt('../data/vertical_mapping_1.csv', vertical_numpy_insert_nan, delimiter=',')


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = DataloaderUi()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
