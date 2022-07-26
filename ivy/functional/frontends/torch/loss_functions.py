import ivy

def _get_reduction_func(reduction):
    if reduction == 'none':
        ret = lambda x : x
    elif reduction == 'mean':
        ret = ivy.mean
    elif reduction == 'elementwise_mean':
        ivy.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = ivy.mean
    elif reduction == 'sum':
        ret = ivy.sum
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret

def _legacy_get_string(size_average, reduce, emit_warning = True):
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

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
    if emit_warning:
        ivy.warn(warning.format(ret))
    return ret

def _get_reduction(reduction, size_average=None, reduce=None, emit_warning = True):
    if size_average is not None or reduce is not None:
        return _get_reduction_func(_legacy_get_string(size_average, reduce, emit_warning))
    else:
        return _get_reduction_func(reduction)

def binary_cross_entropy(
    input, 
    target,
    weight=None, 
    size_average=None, 
    reduce=None, 
    reduction='mean'
):
    reduction = _get_reduction(reduction, size_average, reduce)

    weight = weight.expand(target.shape)
    
    result = reduction(weight*ivy.binary_cross_entropy(target, input, epsilon=0.0))
    
    return result

binary_cross_entropy.unsupported_dtypes = ("float16",)