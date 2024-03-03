import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_device_and_dtypes


@to_ivy_arrays_and_back
def batched_nms(boxes, scores, idxs, iou_threshold):
    if boxes.size == 0:
        return ivy.array([], dtype=ivy.int64)
    else:
        max_coordinate = boxes.max()
        boxes_dtype = boxes.dtype
        offsets = idxs.astype(boxes_dtype) * (
            max_coordinate + ivy.array(1, dtype=boxes_dtype)
        )
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, iou_threshold)
        return keep


@to_ivy_arrays_and_back
def box_area(boxes):
    return ivy.prod(boxes[..., 2:] - boxes[..., :2], axis=-1)


@to_ivy_arrays_and_back
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = ivy.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = ivy.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(x_min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


@with_unsupported_device_and_dtypes(
    {
        "2.2 and below": {
            "cpu": ("float16",),
        }
    },
    "torch",
)
@to_ivy_arrays_and_back
def clip_boxes_to_image(boxes, size):
    height, width = size
    boxes_x = boxes[..., 0::2].clip(0, width)
    boxes_y = boxes[..., 1::2].clip(0, height)
    clipped_boxes = ivy.stack([boxes_x, boxes_y], axis=-1)
    return clipped_boxes.reshape(boxes.shape).astype(boxes.dtype)


@to_ivy_arrays_and_back
def nms(boxes, scores, iou_threshold):
    return ivy.nms(boxes, scores, iou_threshold)


@to_ivy_arrays_and_back
def remove_small_boxes(boxes, min_size):
    w, h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    return ivy.nonzero((w >= min_size) & (h >= min_size))[0]


@with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def roi_align(
    input, boxes, output_size, spatial_scale=1.0, sampling_ratio=1, aligned=False
):
    return ivy.roi_align(
        input, boxes, output_size, spatial_scale, sampling_ratio, aligned
    )
