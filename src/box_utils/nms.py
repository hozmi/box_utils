"""Box NMS."""
import numpy as np

# C extension.
from ._c.box_nms import ltrb_nms, ltwh_nms, xywh_nms


def box_nms(boxes,
            scores,
            score_thresh,
            iou_thresh,
            topk=-1,
            fmt='ltrb'):
    """Apply greedy-NMS."""
    if topk is None:
        topk = -1

    if fmt == 'ltrb':
        return ltrb_nms(
            np.asarray(boxes).astype(np.float32),
            np.asarray(scores).astype(np.float32),
            float(score_thresh),
            float(iou_thresh),
            int(topk)
        )
    if fmt == 'ltwh':
        return ltwh_nms(
            np.asarray(boxes).astype(np.float32),
            np.asarray(scores).astype(np.float32),
            float(score_thresh),
            float(iou_thresh),
            int(topk)
        )
    if fmt == 'xywh':
        return xywh_nms(
            np.asarray(boxes).astype(np.float32),
            np.asarray(scores).astype(np.float32),
            float(score_thresh),
            float(iou_thresh),
            int(topk)
        )
    raise ValueError(f'unknown box format: {fmt}')
