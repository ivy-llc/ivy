# local
import ivy
from ivy import with_supported_dtypes, to_ivy_arrays_and_back


@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def instance_norm(x,
                  running_mean,
                  running_var,
                  weight=None,
                  bias=None,
                  use_input_stats=True,
                  momentum=0.9,
                  eps=1e-05,
                  data_format='NCHW',
                  name=None):
    return ivy.instance_norm(x,
                             scale=None,
                             bias=bias,
                             eps=eps,
                             momentum=momentum,
                             data_format=data_format,
                             running_mean=running_mean,
                             running_stddev=running_var)
