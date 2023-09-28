import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def nms(boxes, scores, iou_threshold):
    return ivy.nms(boxes, scores, iou_threshold)


@to_ivy_arrays_and_back
def roi_align(
    input, boxes, output_size, spatial_scale=1.0, sampling_ratio=1, aligned=False
):
    return ivy.roi_align(
        input, boxes, output_size, spatial_scale, sampling_ratio, aligned
    )
