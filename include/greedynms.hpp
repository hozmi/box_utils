#ifndef _BOX_GREEDYNMS_HPP_
#define _BOX_GREEDYNMS_HPP_

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <vector>


namespace box {


struct ltrb_tag {
    template <typename _Ty>
    static inline _Ty iou(const _Ty a[/*4*/], const _Ty b[/*4*/]) {
        // ltrb
        _Ty a_x1 = std::min(a[0], a[2]);
        _Ty a_y1 = std::min(a[1], a[3]);
        _Ty a_x2 = std::max(a[2], a[0]);
        _Ty a_y2 = std::max(a[3], a[1]);

        _Ty b_x1 = std::min(b[0], b[2]);
        _Ty b_y1 = std::min(b[1], b[3]);
        _Ty b_x2 = std::max(b[2], b[0]);
        _Ty b_y2 = std::max(b[3], b[1]);

        // width/height
        _Ty a_w = a_x2 - a_x1;
        _Ty a_h = a_y2 - a_y1;
        _Ty b_w = b_x2 - b_x1;
        _Ty b_h = b_y2 - b_y1;

        // c = intersection(a, b)
        _Ty c_x1 = std::max(a_x1, b_x1);
        _Ty c_y1 = std::max(a_y1, b_y1);
        _Ty c_x2 = std::min(a_x2, b_x2);
        _Ty c_y2 = std::min(a_y2, b_y2);

        _Ty c_w = std::max((c_x2 - c_x1), static_cast<_Ty>(0));
        _Ty c_h = std::max((c_y2 - c_y1), static_cast<_Ty>(0));

        _Ty a_area = a_w * a_h;
        _Ty b_area = b_w * b_h;
        _Ty c_area = c_w * c_h;

        _Ty iou = c_area / (a_area + b_area - c_area);
        return iou;
    }
};


struct ltwh_tag {
    template <typename _Ty>
    static inline _Ty iou(const _Ty a[/*4*/], const _Ty b[/*4*/])
    {
        // width/height
        _Ty a_w = a[2];
        _Ty a_h = a[3];
        _Ty b_w = b[2];
        _Ty b_h = b[3];

        // ltrb
        _Ty a_x1 = a[0];
        _Ty a_y1 = a[1];
        _Ty a_x2 = a_x1 + a_w;
        _Ty a_y2 = a_y1 + a_h;

        _Ty b_x1 = b[0];
        _Ty b_y1 = b[1];
        _Ty b_x2 = b_x1 + b_w;
        _Ty b_y2 = b_y1 + b_h;

        // c = intersection(a, b)
        _Ty c_x1 = std::max(a_x1, b_x1);
        _Ty c_y1 = std::max(a_y1, b_y1);
        _Ty c_x2 = std::min(a_x2, b_x2);
        _Ty c_y2 = std::min(a_y2, b_y2);

        _Ty c_w = std::max((c_x2 - c_x1), static_cast<_Ty>(0));
        _Ty c_h = std::max((c_y2 - c_y1), static_cast<_Ty>(0));

        _Ty a_area = a_w * a_h;
        _Ty b_area = b_w * b_h;
        _Ty c_area = c_w * c_h;

        _Ty iou = c_area / (a_area + b_area - c_area);
        return iou;
    }
};


struct xywh_tag {
    template <typename _Ty>
    static inline _Ty iou(const _Ty a[/*4*/], const _Ty b[/*4*/])
    {
        // width/height
        _Ty a_w = a[2];
        _Ty a_h = a[3];
        _Ty b_w = b[2];
        _Ty b_h = b[3];

        // ltrb
        _Ty a_x1 = a[0] - a_w / 2;
        _Ty a_y1 = a[1] - a_h / 2;
        _Ty a_x2 = a[0] + a_w / 2;
        _Ty a_y2 = a[1] + a_h / 2;

        _Ty b_x1 = b[0] - b_w / 2;
        _Ty b_y1 = b[1] - b_h / 2;
        _Ty b_x2 = b[0] + b_w / 2;
        _Ty b_y2 = b[1] + b_h / 2;

        // c = intersection(a, b)
        _Ty c_x1 = std::max(a_x1, b_x1);
        _Ty c_y1 = std::max(a_y1, b_y1);
        _Ty c_x2 = std::min(a_x2, b_x2);
        _Ty c_y2 = std::min(a_y2, b_y2);

        _Ty c_w = std::max((c_x2 - c_x1), static_cast<_Ty>(0));
        _Ty c_h = std::max((c_y2 - c_y1), static_cast<_Ty>(0));

        _Ty a_area = a_w * a_h;
        _Ty b_area = b_w * b_h;
        _Ty c_area = c_w * c_h;

        _Ty iou = c_area / (a_area + b_area - c_area);
        return iou;
    }
};


template <typename _Ty, typename _Box>
inline _Ty iou(const _Ty a[/*4*/], const _Ty b[/*4*/], _Box)
{
    return _Box::iou(a, b);
}


template <typename _Ty, typename _Int, typename _Box>
inline _Ty max_iou(const _Ty box[/*4*/],
                   const _Ty boxes[/*n-by-4*/],
                   const _Int keep[/*num*/], _Int num,
                   _Box)
{
    const _Int ldb = 4;
    _Ty maxv = 0;
    for (_Int i=0; i<num; ++i) {
        _Int index = keep[i];
        _Ty v = _Box::iou(box, boxes + index * ldb);
        if (maxv < v) {
            maxv = v;
        }
    }
    return maxv;
}


/**
    @brief Apply greedy NMS.
    @tparam _Ty  real type.
    @tparam _Int index type.
    @tparam _Box tag-dispatch.
    @param[in] n_boxes number of boxes.
    @param[in] boxes   bounding boxes of objects, shape=(n_boxes, 4).
    @param[in] scores  confidence score of each object.
    @param[out] keep   selected indices from the boxes, array[n_boxes].
    @param[in] score_thresh threshold for the scores.
    @param[in] iou_thresh   threshold for IoU.
    @param[in] fmt     tag-dispatch.
    @return number of the selected boxes.
 */
template <typename _Ty, typename _Int, typename _Box=ltrb_tag>
_Int greedy_nms(_Int n_boxes,
                const _Ty boxes[/*n-by-4*/],
                const _Ty scores[/*n*/],
                _Int keep[/*n*/],
                _Ty score_thresh, _Ty iou_thresh,
                _Box fmt={})
{
    const _Int ldb = 4;
    auto comp = [&scores](_Int a, _Int b) -> bool {
        _Ty sa = scores[a];
        _Ty sb = scores[b];
        return (sa < sb) || (sa == sb && a > b);
    };

    _Int remainder = 0;
    _Int* order = keep;

    for (_Int i=0; i<n_boxes; ++i) {
        if (scores[i] < score_thresh) { continue; }
        order[remainder] = i;
        ++remainder;
    }
    if (remainder <= 1) { return remainder; }

    std::reverse_iterator<_Int*> top(order + remainder);
    std::reverse_iterator<_Int*> end(order);
    std::make_heap(top, end, comp);  // priority queue

    _Int n_keep = 0;
    while (remainder) {
        _Int index = *top;
        const _Ty* pbox = boxes + index * ldb;
        std::pop_heap(top, end, comp);
        --end;
        --remainder;
        if (max_iou(pbox, boxes, keep, n_keep, fmt) <= iou_thresh) {
            keep[n_keep] = index;
            ++n_keep;
        }
    }
    return n_keep;
}


/**
    @brief Apply greedy NMS.
    @tparam _Ty  real type.
    @tparam _Int index type.
    @tparam _Box tag-dispatch.
    @param[in] n_boxes number of boxes.
    @param[in] boxes   bounding boxes of objects, shape=(n_boxes, 4).
    @param[in] scores  confidence score of each object.
    @param[out] keep   selected indices from the boxes, array[n_boxes].
    @param[in] score_thresh threshold for the scores.
    @param[in] iou_thresh   threshold for IoU.
    @param[in] topk    maximum number of the boxes to be selected.
    @param[in] fmt     tag-dispatch.
    @return number of the selected boxes.
 */
template <typename _Ty, typename _Int, typename _Box=ltrb_tag>
_Int greedy_nms(_Int n_boxes,
                const _Ty boxes[/*n-by-4*/],
                const _Ty scores[/*n*/],
                _Int keep[/*n*/],
                _Ty score_thresh, _Ty iou_thresh, _Int topk,
                _Box fmt={})
{
    const _Int ldb = 4;
    auto comp = [&scores](_Int a, _Int b) -> bool {
        _Ty sa = scores[a];
        _Ty sb = scores[b];
        return (sa < sb) || (sa == sb && a > b);
    };

    _Int remainder = 0;
    _Int* order = keep;

    if (topk <= 0) { return 0; }
    for (_Int i=0; i<n_boxes; ++i) {
        if (scores[i] < score_thresh) { continue; }
        order[remainder] = i;
        ++remainder;
    }
    if (remainder <= 1) { return remainder; }

    std::reverse_iterator<_Int*> top(order + remainder);
    std::reverse_iterator<_Int*> end(order);
    std::make_heap(top, end, comp);  // priority queue

    _Int n_keep = 0;
    while (remainder) {
        _Int index = *top;
        const _Ty* pbox = boxes + index * ldb;
        std::pop_heap(top, end, comp);
        --end;
        --remainder;
        if (max_iou(pbox, boxes, keep, n_keep, fmt) <= iou_thresh) {
            keep[n_keep] = index;
            ++n_keep;
            if (n_keep >= topk) { break; }
        }
    }
    return n_keep;
}


/**
    @brief Apply greedy NMS.
    @tparam _Ty  real type.
    @tparam _Int index type.
    @tparam _Alloc allocator for a vector.
    @tparam _Box tag-dispatch.
    @param[in] n_classes number of classes, @em k.
    @param[in] n_boxes number of boxes, @em n.
    @param[in] boxes   bounding boxes of objects, shape=(n, 4).
    @param[in] scores  confidence score of each object, shape=(k, n).
    @param[out] keep   selected indices from the boxes.
    @param[in] score_thresh threshold for the scores.
    @param[in] iou_thresh   threshold for IoU.
    @param[in] topk    maximum number of the boxes to be selected.
    @param[in] fmt     tag-dispatch.
    @exception std::bad_alloc if failed to allocate buffers.
 */
template <
    typename _Ty, typename _Int, typename _Alloc, typename _Box=ltrb_tag>
void greedy_nms(_Int n_classes, _Int n_boxes,
                const _Ty boxes[/*n-by-4*/],
                const _Ty scores[/*k-by-n*/],
                std::vector<std::array<_Int, 2>, _Alloc>& keep,
                _Ty score_thresh, _Ty iou_thresh, _Int topk,
                _Box fmt={})
{
    std::unique_ptr<_Int[]> buffer(new _Int[n_boxes]);
    if (topk < 0) {
        for (_Int j=0; j < n_classes; ++j) {
            _Int n_keep = greedy_nms(
                n_boxes, boxes, scores + n_boxes * j,
                buffer.get(), score_thresh, iou_thresh, fmt);

            keep.reserve(keep.size() + n_keep);
            for (_Int i=0; i < n_keep; ++i) {
                keep.push_back({j, buffer[i]});
            }
        }
    }
    else {
        for (_Int j=0; j < n_classes; ++j) {
            _Int n_keep = greedy_nms(
                n_boxes, boxes, scores + n_boxes * j,
                buffer.get(), score_thresh, iou_thresh, topk, fmt);

            keep.reserve(keep.size() + n_keep);
            for (_Int i=0; i < n_keep; ++i) {
                keep.push_back({j, buffer[i]});
            }
        }
    }
}


/**
    @brief Apply greedy NMS.
    @tparam _Ty  real type.
    @tparam _Int index type.
    @tparam _Alloc allocator for a vector.
    @tparam _Box tag-dispatch.
    @param[in] n_batches number of batches, @em b.
    @param[in] n_classes number of classes, @em k.
    @param[in] n_boxes number of boxes, @em n.
    @param[in] boxes   bounding boxes of objects, shape=(b, n, 4).
    @param[in] scores  confidence score of each object, shape=(b, k, n).
    @param[out] keep   selected indices from the boxes.
    @param[in] score_thresh threshold for the scores.
    @param[in] iou_thresh   threshold for IoU.
    @param[in] topk    maximum number of the boxes to be selected.
    @param[in] fmt     tag-dispatch.
    @exception std::bad_alloc if failed to allocate buffers.
 */
template <
    typename _Ty, typename _Int, typename _Alloc, typename _Box=ltrb_tag>
void greedy_nms(_Int n_batches, _Int n_classes, _Int n_boxes,
                const _Ty boxes[/*(b, n, 4)*/],
                const _Ty scores[/*(b, k, n)*/],
                std::vector<std::array<_Int, 3>, _Alloc>& keep,
                _Ty score_thresh, _Ty iou_thresh, _Int topk,
                _Box fmt={})
{
    const _Int ldb = 4;
    std::unique_ptr<_Int[]> buffer(new _Int[n_boxes]);
    if (topk < 0) {
        for (_Int k=0; k < n_batches; ++k) {
            const _Ty* b_boxes = boxes + n_boxes * ldb * k;
            const _Ty* b_scores = scores + n_classes * n_boxes * k;

            for (_Int j=0; j < n_classes; ++j) {
                _Int n_keep = greedy_nms(
                    n_boxes, b_boxes, b_scores + n_boxes * j,
                    buffer.get(), score_thresh, iou_thresh, fmt);

                keep.reserve(keep.size() + n_keep);
                for (_Int i=0; i < n_keep; ++i) {
                    keep.push_back({k, j, buffer[i]});
                }
            }
        }
    }
    else {
        for (_Int k=0; k < n_batches; ++k) {
            const _Ty* b_boxes = boxes + n_boxes * ldb * k;
            const _Ty* b_scores = scores + n_classes * n_boxes * k;

            for (_Int j=0; j < n_classes; ++j) {
                _Int n_keep = greedy_nms(
                    n_boxes, b_boxes, b_scores + n_boxes * j,
                    buffer.get(), score_thresh, iou_thresh, topk, fmt);

                keep.reserve(keep.size() + n_keep);
                for (_Int i=0; i < n_keep; ++i) {
                    keep.push_back({k, j, buffer[i]});
                }
            }
        }
    }
}


}  // namespace box


#endif
