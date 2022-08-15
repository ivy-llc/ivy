# global
import ivy


def _get_reduction_func(reduction):    
    if reduction == 'none':
        ret = lambda x : x
    elif reduction == 'mean':
        ret = ivy.mean
    elif reduction == 'elementwise_mean':
        ret = ivy.mean
    elif reduction == 'sum':
        ret = ivy.sum
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret


def _legacy_get_string(size_average, reduce):
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    return ret


def _get_reduction(reduction, 
                   size_average=None, 
                   reduce=None):
    if size_average is not None or reduce is not None:
        return _get_reduction_func(_legacy_get_string(size_average, reduce))
    else:
        return _get_reduction_func(reduction)


def cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    return ivy.cross_entropy(input, target)
 

cross_entropy.unsupported_dtypes = ('uint16', 'float16', 'uint64', 'uint32')


def binary_cross_entropy(
    input, 
    target, 
    weight=None, 
    size_average=None, 
    reduce=None, 
    reduction='mean'
):
    reduction = _get_reduction(reduction, size_average, reduce)
    result = ivy.binary_cross_entropy(target, input, epsilon=0.0)
    
    if weight is not None:
        result = ivy.multiply(weight, result)
    result = reduction(result)
    return result


binary_cross_entropy.unsupported_dtypes = (
    'uint16', 
    'float16', 
    'uint64', 
    'float64', 
    'uint32'
)
