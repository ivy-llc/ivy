# local
import ivy
from ivy import with_supported_dtypes, to_ivy_arrays_and_back


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def instance_norm(
        x,
        mean,
        variance,
        /,
        *,
        offset=None,
        scale=None,
        eps=1e-05,
        momentum=0.9,
        data_format="NCH",
        training=False,
        name=None,
):
    return ivy.instance_norm(x, mean, variance,
                                               eps=eps, momentum=momentum,
                                               offset=offset, scale=scale,
                                               data_format=data_format, training=training)
