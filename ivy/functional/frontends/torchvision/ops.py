import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@to_ivy_arrays_and_back
def nms(boxes, scores, iou_threshold):
    return ivy.nms(boxes, scores, iou_threshold)


@with_supported_dtypes({"2.1.0 and below": ("float32", "float64")}, "torch")
@to_ivy_arrays_and_back
def roi_align(
    input, boxes, output_size, spatial_scale=1.0, sampling_ratio=1, aligned=False
):
    return ivy.roi_align(
        input, boxes, output_size, spatial_scale, sampling_ratio, aligned
    )
