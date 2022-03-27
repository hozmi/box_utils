"""Box format converter."""
import numpy as np


def ltrb_to_ltrb(box):
    """Convert 'ltrb' to 'ltwrb'."""
    return box


def ltrb_to_ltwh(box):
    """Convert 'ltrb' to 'ltwh'."""
    left = box[..., 0]
    top = box[..., 1]
    right = box[..., 2]
    bottom = box[..., 3]

    width = right - left
    height = bottom - top

    return np.stack((left, top, width, height), axis=-1)


def ltrb_to_xywh(box):
    """Convert 'ltrb' to 'xywh'."""
    left = box[..., 0]
    top = box[..., 1]
    right = box[..., 2]
    bottom = box[..., 3]

    width = right - left
    height = bottom - top
    center_x = left + width /2
    center_y = top + height / 2

    return np.stack((center_x, center_y, width, height), axis=-1)
