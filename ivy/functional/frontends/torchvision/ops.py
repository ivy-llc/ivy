import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_device_and_dtypes


@to_ivy_arrays_and_back
def box_area(boxes):
    return ivy.prod(boxes[..., 2:] - boxes[..., :2], axis=-1)


@with_unsupported_device_and_dtypes(
    {
        "2.1.0 and below": {
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


@with_supported_dtypes({"2.1.0 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def roi_align(
    input, boxes, output_size, spatial_scale=1.0, sampling_ratio=1, aligned=False
):
    return ivy.roi_align(
        input, boxes, output_size, spatial_scale, sampling_ratio, aligned
    )
