# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def cosine_similarity(x1, x2, *, axis=1, eps=1e-08):
    if len(x1.shape) == len(x2.shape) and len(x2.shape) >= 2:
        numerator = ivy.sum(x1 * x2, axis=axis)
        x1_squared_norm = ivy.sum(ivy.square(x1), axis=axis)
        x2_squared_norm = ivy.sum(ivy.square(x2), axis=axis)
    else:
        numerator = ivy.sum(x1 * x2)
        x1_squared_norm = ivy.sum(ivy.square(x1))
        x2_squared_norm = ivy.sum(ivy.square(x2))

    x1_norm = ivy.sqrt(x1_squared_norm)
    x2_norm = ivy.sqrt(x2_squared_norm)
    norm_mm = x1_norm * x2_norm
    denominator = ivy.maximum(norm_mm, eps)

    cosine = numerator / denominator
    return cosine


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def dropout2d(x, *, p=0.5):
    if not isinstance(x, ivy.Array):
        raise TypeError(f'Input x must be an instance of {ivy.backend} Tensor but was {type(x)}')
    if p < 0 or p > 1:
        raise ValueError(f'Input p must be a value between 0 and 1, but was {p}')
    
    if x.shape != (None, 4, None, None):
        raise ValueError(f'Input x must be a 4D Tensor but was {x.shape}')
    
    if p == 0:
        return x
    
    mask =  ivy.random.choice([0, 1], size=x.shape, p=[p, 1 - p])
    mask /= (1 - p)
    return x * mask