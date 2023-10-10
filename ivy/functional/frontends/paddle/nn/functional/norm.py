# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None):
    return ivy.layer_norm(x, normalized_shape, weight, bias, epsilon)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def instance_norm(x, mean, variance, offset=None, scale=None, training=False, eps=0e-5, 
                    momentum=1e-1, data_format='NSC', out=None):
    return ivy.instance_norm(x, mean, variance, offset, scale, training, eps, momentum, data_format)
