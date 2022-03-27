#include <stdint.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nmsmodule.hpp"


template <typename _Ty, typename _Int>
py::array_t<_Int, py::array::c_style> ltrb_nms(
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> scores,
    _Ty score_thresh, _Ty iou_thresh, _Int topk=-1
) {
    return box::py_greedy_nms(
        boxes, scores, score_thresh, iou_thresh, topk, box::ltrb_tag{});
}


template <typename _Ty, typename _Int>
py::array_t<_Int, py::array::c_style> ltwh_nms(
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> scores,
    _Ty score_thresh, _Ty iou_thresh, _Int topk=-1
) {
    return box::py_greedy_nms(
        boxes, scores, score_thresh, iou_thresh, topk, box::ltwh_tag{});
}


template <typename _Ty, typename _Int>
py::array_t<_Int, py::array::c_style> xywh_nms(
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<_Ty, py::array::c_style | py::array::forcecast> scores,
    _Ty score_thresh, _Ty iou_thresh, _Int topk=-1
) {
    return box::py_greedy_nms(
        boxes, scores, score_thresh, iou_thresh, topk, box::xywh_tag{});
}


PYBIND11_MODULE(box_nms, m) {
  m.doc() = "Box NMS.";
  m.def("ltrb_nms", &ltrb_nms<float, intptr_t>,
    "Apply greedy NMS.");

  m.def("ltwh_nms", &ltwh_nms<float, intptr_t>,
    "Apply greedy NMS.");

  m.def("xywh_nms", &xywh_nms<float, intptr_t>,
    "Apply greedy NMS.");

}
