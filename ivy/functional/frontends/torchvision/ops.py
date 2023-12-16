import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_device_and_dtypes


@to_ivy_arrays_and_back
def batched_nms(boxes, scores, idxs, iou_threshold):
    """Perform non-maximum suppression (NMS) on a batch of bounding boxes.

    Parameters:
    - boxes (array): Bounding box coordinates.
    - scores (array): Confidence scores for each box.
    - idxs (array): Indices of the boxes in the batch.
    - iou_threshold (float): Intersection over union threshold for NMS.

    Returns:
    - array: Indices of the selected boxes after NMS.
    """
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
    """Calculate the area of each bounding box.

    Parameters:
    - boxes (array): Bounding box coordinates.

    Returns:
    - array: Array containing the area of each bounding box.
    """
    return ivy.prod(boxes[..., 2:] - boxes[..., :2], axis=-1)


@with_unsupported_device_and_dtypes(
    {
        "2.1.1 and below": {
            "cpu": ("float16",),
        }
    },
    "torch",
)
@to_ivy_arrays_and_back
def clip_boxes_to_image(boxes, size):
    """Clip bounding boxes to be within the image dimensions.

    Parameters:
    - boxes (array): Bounding box coordinates.
    - size (tuple): Height and width of the image.

    Returns:
    - array: Clipped bounding box coordinates.
    """
    height, width = size
    boxes_x = boxes[..., 0::2].clip(0, width)
    boxes_y = boxes[..., 1::2].clip(0, height)
    clipped_boxes = ivy.stack([boxes_x, boxes_y], axis=-1)
    return clipped_boxes.reshape(boxes.shape).astype(boxes.dtype)


@to_ivy_arrays_and_back
def nms(boxes, scores, iou_threshold):
    """Perform non-maximum suppression (NMS) on bounding boxes.

    Parameters:
    - boxes (array): Bounding box coordinates.
    - scores (array): Confidence scores for each box.
    - iou_threshold (float): Intersection over union threshold for NMS.

    Returns:
    - array: Indices of the selected boxes after NMS.
    """
    return ivy.nms(boxes, scores, iou_threshold)


@to_ivy_arrays_and_back
def remove_small_boxes(boxes, min_size):
    """Remove small bounding boxes based on a minimum size threshold.

    Parameters:
    - boxes (array): Bounding box coordinates.
    - min_size (float): Minimum size threshold.

    Returns:
    - array: Indices of the remaining boxes after size filtering.
    """
    w, h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    return ivy.nonzero((w >= min_size) & (h >= min_size))[0]


@with_supported_dtypes({"2.1.1 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def roi_align(
    input, boxes, output_size, spatial_scale=1.0, sampling_ratio=1, aligned=False
):
    """Perform Region of Interest (ROI) pooling or align pooling.

    Parameters:
    - input (array): Input tensor.
    - boxes (array): Bounding box coordinates.
    - output_size (tuple): Size of the output after pooling.
    - spatial_scale (float): Spatial scale factor.
    - sampling_ratio (int): Sampling ratio for ROI align.
    - aligned (bool): Whether to use aligned ROI pooling.

    Returns:
    - array: Result of ROI pooling or align operation.
    """
    return ivy.roi_align(
        input, boxes, output_size, spatial_scale, sampling_ratio, aligned
    )
