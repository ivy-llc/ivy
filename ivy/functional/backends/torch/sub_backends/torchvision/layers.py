from torchvision.ops import roi_align as torch_roi_align
from ivy.func_wrapper import to_native_arrays_and_back


@to_native_arrays_and_back
def roi_align(
    input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False
):
    ret = torch_roi_align(
        input, boxes, output_size, spatial_scale, sampling_ratio, aligned
    )
    return ret
