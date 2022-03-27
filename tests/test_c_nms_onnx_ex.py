"""Test NMS.

Run the examples described in `ONNX docs`_.

.. _ONNX docs: https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression
"""
# import pytest
import numpy as np

import box_utils._c.box_nms as box_nms


def test_nms_suppress_by_iou():
    """Test NMS - suppress by IoU."""
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

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_suppress_by_IOU_and_scores():
    """Test NMS - suppress by IoU and scores."""
    # --
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    scores = np.array(
        [[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.4]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_single_box():
    """Test NMS - single box."""
    # --
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_identical_boxes():
    """Test NMS - identical boxes."""
    # --
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],

        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ]]).astype(np.float32)
    scores = np.array([[[
        0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9
    ]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 0]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_limit_output_size():
    """Test NMS - limit output size."""
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
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_two_batches():
    """Test NMS - two batches."""
    # --
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, -0.1, 1.0, 0.9],
                    [0.0, 10.0, 1.0, 11.0],
                    [0.0, 10.1, 1.0, 11.1],
                    [0.0, 100.0, 1.0, 101.0]],
                    [[0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, -0.1, 1.0, 0.9],
                    [0.0, 10.0, 1.0, 11.0],
                    [0.0, 10.1, 1.0, 11.1],
                    [0.0, 100.0, 1.0, 101.0]]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
                    [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([
        [0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_two_classes():
    """Test NMS - two classes."""
    # --
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    scores = np.array([[
        [0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
        [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([
        [0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_center_point_box_format():
    """Test NMS - center-point box format."""
    # --
    boxes = np.array([[
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.6, 1.0, 1.0],
        [0.5, 0.4, 1.0, 1.0],
        [0.5, 10.5, 1.0, 1.0],
        [0.5, 10.6, 1.0, 1.0],
        [0.5, 100.5, 1.0, 1.0]
    ]]).astype(np.float32)
    scores = np.array([[
        [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([
        [0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
    # --

    result = box_nms.xywh_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_flipped_coordinates():
    """Test NMS - flipped coordinates."""
    # --
    boxes = np.array([[
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, 0.9, 1.0, -0.1],
        [0.0, 10.0, 1.0, 11.0],
        [1.0, 10.1, 0.0, 11.1],
        [1.0, 101.0, 0.0, 100.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


# ---------------------------------------------------------
# box_nms can be called in some other way.
# ---------------------------------------------------------

def test_nms_suppress_by_iou_nobatch():
    """Test NMS - suppress by IoU."""
    # --
    boxes = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]).astype(np.float32)
    scores = np.array([[
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array(
        [[0, 3], [0, 0], [0, 5]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_suppress_by_iou_noclass():
    """Test NMS - suppress by IoU."""
    # --
    boxes = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]).astype(np.float32)
    scores = np.array([
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([3, 0, 5]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_suppress_by_iou_notopk():
    """Test NMS - suppress by IoU."""
    # --
    boxes = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]).astype(np.float32)
    scores = np.array([
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
    max_output_boxes_per_class = np.array([-1]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([3, 0, 5]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_two_classes_nobatch():
    """Test NMS - two classes."""
    # --
    boxes = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]).astype(np.float32)
    scores = np.array([
        [0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
        [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([
        [0, 3], [0, 0], [1, 3], [1, 0]]).astype(np.int64)
    # --

    result = box_nms.ltrb_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_center_point_box_format_nobatch():
    """Test NMS - center-point box format."""
    # --
    boxes = np.array([
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.6, 1.0, 1.0],
        [0.5, 0.4, 1.0, 1.0],
        [0.5, 10.5, 1.0, 1.0],
        [0.5, 10.6, 1.0, 1.0],
        [0.5, 100.5, 1.0, 1.0]
    ]).astype(np.float32)
    scores = np.array([
        [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([
        [0, 3], [0, 0], [0, 5]]).astype(np.int64)
    # --

    result = box_nms.xywh_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)


def test_nms_center_point_box_format_noclass():
    """Test NMS - center-point box format."""
    # --
    boxes = np.array([
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.6, 1.0, 1.0],
        [0.5, 0.4, 1.0, 1.0],
        [0.5, 10.5, 1.0, 1.0],
        [0.5, 10.6, 1.0, 1.0],
        [0.5, 100.5, 1.0, 1.0]
    ]).astype(np.float32)
    scores = np.array(
        [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([3, 0, 5]).astype(np.int64)
    # --

    result = box_nms.xywh_nms(
        boxes, scores,
        score_threshold[0], iou_threshold[0], max_output_boxes_per_class[0])

    np.testing.assert_array_equal(result, selected_indices)
