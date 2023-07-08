# %%
import cv2

def fitting_ellipse(point_location) -> (float, float, float):
    return cv2.fitEllipseAMS(point_location)