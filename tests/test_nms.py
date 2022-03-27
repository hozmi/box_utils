"""Test NMS."""
import pytest
import numpy as np

import box_utils.converter
from box_utils.nms import box_nms


@pytest.mark.parametrize("fmt, convert", [
    ("ltrb", box_utils.converter.ltrb_to_ltrb),
    ("ltwh", box_utils.converter.ltrb_to_ltwh),
    ("xywh", box_utils.converter.ltrb_to_xywh),
])
def test_nms_suppress_by_iou(fmt, convert):
    """Test NMS - suppress by IoU."""
    # example from https://github.com/onnx/onnx/
    # --
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    scores = np.array([[[
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array(
        [[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
    # --

    result = box_nms(
        convert(boxes), scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0],
        fmt)

    np.testing.assert_array_equal(result, selected_indices)
