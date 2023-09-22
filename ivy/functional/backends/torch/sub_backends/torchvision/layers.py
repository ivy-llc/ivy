from torchvision.ops import nms as torch_nms
import torch


def nms(
    boxes,
    scores=None,
    iou_threshold=0.5,
    max_output_size=None,
    score_threshold=float("-inf"),
):
    # boxes (Tensor[N, 4])) – boxes to perform NMS on.
    # They are expected to be in (x1, y1, x2, y2) format
    # with 0 <= x1 < x2 and 0 <= y1 < y2.
    change_id = False
    if score_threshold is not float("-inf") and scores is not None:
        keep_idx = scores > score_threshold
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        change_id = True
        nonzero = torch.nonzero(keep_idx).flatten()

    if scores is None:
        scores = torch.ones((boxes.shape[0],), dtype=boxes.dtype)

    if len(boxes) < 2:
        if len(boxes) == 1:
            ret = torch.tensor([0], dtype=torch.int64)
        else:
            ret = torch.tensor([], dtype=torch.int64)
    else:
        ret = torch_nms(boxes, scores, iou_threshold)

    if change_id and len(ret) > 0:
        ret = torch.tensor(nonzero[ret], dtype=torch.int64).flatten()

    return ret.flatten()[:max_output_size]
