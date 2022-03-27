#ifndef _BOX_NMSMODULE_HPP_
#define _BOX_NMSMODULE_HPP_

#include <array>
#include <memory>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "greedynms.hpp"


namespace py = pybind11;


namespace box {


template <typename _Ty, typename _Int, typename _Box=ltrb_tag>
py::array_t<_Int, py::array::c_style> py_greedy_nms(
    _Int n_boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> scores,
    _Ty score_thresh, _Ty iou_thresh, _Int topk,
    _Box fmt={}
) {
    py::buffer_info b_boxes = boxes.request();
    py::buffer_info b_scores = scores.request();

    _Int n_keep = 0;
    std::unique_ptr<_Int[]> tmp(new _Int[n_boxes]);
    if (topk < 0) {
        n_keep = greedy_nms(
            n_boxes,
            static_cast<_Ty*>(b_boxes.ptr),
            static_cast<_Ty*>(b_scores.ptr),
            tmp.get(),
            score_thresh, iou_thresh,
            fmt
        );
    }
    else {
        n_keep = greedy_nms(
            n_boxes,
            static_cast<_Ty*>(b_boxes.ptr),
            static_cast<_Ty*>(b_scores.ptr),
            tmp.get(),
            score_thresh, iou_thresh, topk,
            fmt
        );
    }

    py::array_t<_Int, py::array::c_style> keep;
    keep.resize({n_keep});
    py::buffer_info b_keep = keep.request();
    _Int* p_keep = static_cast<_Int*>(b_keep.ptr); 
    for (_Int i=0; i < n_keep; ++i) {
        p_keep[i] = tmp[i];
    }

    return keep;
}


template <typename _Ty, typename _Int, typename _Box=ltrb_tag>
py::array_t<_Int, py::array::c_style> py_greedy_nms(
    _Int n_classes, _Int n_boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> scores,
    _Ty score_thresh, _Ty iou_thresh, _Int topk,
    _Box fmt={}
) {
    py::buffer_info b_boxes = boxes.request();
    py::buffer_info b_scores = scores.request();

    std::vector<std::array<_Int, 2>> tmp;
    greedy_nms(n_classes, n_boxes,
               static_cast<_Ty*>(b_boxes.ptr),
               static_cast<_Ty*>(b_scores.ptr),
               tmp,
               score_thresh, iou_thresh, topk,
               fmt);

    _Int n_keep = static_cast<_Int>(tmp.size());
    py::array_t<_Int, py::array::c_style> keep;
    keep.resize({n_keep, static_cast<_Int>(2)});
    py::buffer_info b_keep = keep.request();
    _Int* p_keep = static_cast<_Int*>(b_keep.ptr); 
    for (_Int i=0; i < n_keep; ++i) {
        p_keep[2 * i + 0] = tmp[i][0];
        p_keep[2 * i + 1] = tmp[i][1];
    }

    return keep;
}


template <typename _Ty, typename _Int, typename _Box=ltrb_tag>
py::array_t<_Int, py::array::c_style> py_greedy_nms(
    _Int n_batches, _Int n_classes, _Int n_boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> scores,
    _Ty score_thresh, _Ty iou_thresh, _Int topk,
    _Box fmt={}
) {
    py::buffer_info b_boxes = boxes.request();
    py::buffer_info b_scores = scores.request();

    std::vector<std::array<_Int, 3>> tmp;
    greedy_nms(n_batches, n_classes, n_boxes,
               static_cast<_Ty*>(b_boxes.ptr),
               static_cast<_Ty*>(b_scores.ptr),
               tmp,
               score_thresh, iou_thresh, topk,
               fmt);

    _Int n_keep = static_cast<_Int>(tmp.size());
    py::array_t<_Int, py::array::c_style> keep;
    keep.resize({n_keep, static_cast<_Int>(3)});
    py::buffer_info b_keep = keep.request();
    _Int* p_keep = static_cast<_Int*>(b_keep.ptr); 
    for (_Int i=0; i < n_keep; ++i) {
        p_keep[3 * i + 0] = tmp[i][0];
        p_keep[3 * i + 1] = tmp[i][1];
        p_keep[3 * i + 2] = tmp[i][2];
    }

    return keep;
}


template <typename _Ty, typename _Int, typename _Box=ltrb_tag>
py::array_t<_Int, py::array::c_style> py_greedy_nms(
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> scores,
    _Ty score_thresh, _Ty iou_thresh, _Int topk,
    _Box fmt={}
) {
    py::buffer_info b_boxes = boxes.request();
    py::buffer_info b_scores = scores.request();

    if (b_scores.ndim == 3 && b_boxes.ndim == 3) {
        _Int n_batches = b_scores.shape[0];
        _Int n_classes = b_scores.shape[1];
        _Int n_boxes   = b_scores.shape[2];
        if (n_batches != b_boxes.shape[0]) {
            throw std::invalid_argument(
                "Batch sizes are not compatible between boxes and scores.");
        }
        if (n_boxes != b_boxes.shape[1]) {
            throw std::invalid_argument(
                "Number of boxes are not compatible to scores.");
        }
        return py_greedy_nms(
            n_batches, n_classes, n_boxes,
            boxes, scores,
            score_thresh, iou_thresh, topk, fmt);
    }
    else if (b_scores.ndim == 2 && b_boxes.ndim == 2) {
        _Int n_classes = b_scores.shape[0];
        _Int n_boxes   = b_scores.shape[1];
        if (n_boxes != b_boxes.shape[0]) {
            throw std::invalid_argument(
                "Number of boxes are not compatible to scores.");
        }
        return py_greedy_nms(
            n_classes, n_boxes,
            boxes, scores,
            score_thresh, iou_thresh, topk, fmt);
    }
    else if (b_scores.ndim == 1 && b_boxes.ndim == 2) {
        _Int n_boxes   = b_scores.shape[0];
        if (n_boxes != b_boxes.shape[0]) {
            throw std::invalid_argument(
                "Number of boxes are not compatible to scores.");
        }
        return py_greedy_nms(
            n_boxes,
            boxes, scores,
            score_thresh, iou_thresh, topk, fmt);
    }
    throw std::invalid_argument(
        "Shape of scores should be (batches, classes, boxes).");
}


}  // namespace box


#endif
