# %%
import sys

from PySide6 import QtCore, QtWidgets, QtGui
from src.Dataloader import Dataloader
import matplotlib.pyplot as plt
from PIL import Image, ImageQt
import numpy as np
from utils.dijkstra import shortest_path


def open_image_by_plt(image_data: np.array):
    plt.imshow(image_data)
    plt.show()


class DataloaderUi(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.shortest_path_list = None
        self.anchor_list = None
        self.file_name = None
        self.dataloader = Dataloader()
        self.center_x = None
        self.center_y = None
        self.image_navigation_list = []

        self.button_select_file = QtWidgets.QPushButton("Click for select file")
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

        # >>>>>  Layout setting  >>>>>
        self.button_select_file.clicked.connect(self.open_file_list)
        self.button_popup_plt_to_selection_center.clicked.connect(self.popup_plt_image)
        self.button_transform_coordinate_cartesian_to_polar.clicked.connect(
            self.transform_coordinate_cartesian_to_polar)
        self.button_set_anchor.clicked.connect(self.open_reside_image_for_anchor)
        self.button_find_curve_polar.clicked.connect(self.find_path)

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
        new_center = [int(self.center_x * new_width // self.dataloader.original_image.shape[0]), int(self.center_y)]
        print(new_center)
        self.dataloader.transform_car_to_polar(
            center=[new_center[0], new_center[1]],
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
                canva[x, y] = 0.2
        plt.imshow(canva + self.dataloader.polar_image)
        plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = DataloaderUi()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
