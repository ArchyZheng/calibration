import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from PIL import Image
import numpy as np
from matplotlib.backend_bases import MouseEvent


def get_middle(img_data: str) -> (int, int):
    """
    manually choose the middle point of the image.
    :param img_data: the data of the image
    """
    middle = []

    def on_click(event: MouseEvent):
        middle.append([event.xdata, event.ydata])

    fig, ax = plt.subplots()
    ax.imshow(img_data)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    return int(middle[-1][0]), int(middle[-1][1])
