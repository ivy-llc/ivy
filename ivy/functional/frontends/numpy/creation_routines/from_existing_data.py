import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
    handle_numpy_out
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def array(object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    return ivy.array(object, copy=copy, dtype=dtype)


@handle_numpy_dtype
@to_ivy_arrays_and_back
def asarray(
    a,
    dtype=None,
    order=None,
    *,
    like=None,
):
    return ivy.asarray(a, dtype=dtype)


@to_ivy_arrays_and_back
def copy(a, order="K", subok=False):
    return ivy.copy_array(a)


def check_shapes_numpy_broadcastable(shape1, shape2):
    """ Checks whether two shapes satisfy Numpy's broadcasting rules. """
    for s1, s2 in zip(shape1[::-1], shape2[::-1]):
        if s1 != 1 and s2 != 1 and s1 != s2:
            return False
    return True
        
    
def numpy_style_broadcast(a1, a2):
    """ Broadcast two arrays as Numpy would do it. """
    assert check_shapes_numpy_broadcastable(a1.shape, a2.shape), \
        f"Could not broadcast shapes {a1.shape} and {a2.shape}"
    if a1.ndim > a2.ndim:
        a2_ext_shape = (a1.ndim - a2.ndim) * (1,) + a2.shape
        a1_ext_shape = a1.shape
    elif a2.ndim > a1.ndim:
        a1_ext_shape = (a2.ndim - a1.ndim) * (1,) + a1.shape
        a2_ext_shape = a2.shape
    else:
        a1_ext_shape = a1.shape
        a2_ext_shape = a2.shape
    final_shape = ivy.maximum(a1_ext_shape, a2_ext_shape)
    a1_tile_reps = (int(fd - ad) + 1 for fd, ad in zip(final_shape, a1_ext_shape))
    a2_tile_reps = (int(fd - ad) + 1 for fd, ad in zip(final_shape, a2_ext_shape))

    return a1.tile(a1_tile_reps), a2.tile(a2_tile_reps)


@handle_numpy_out
@to_ivy_arrays_and_back
def choose(a, choices, out=None, mode='raise'):
    print("Called with")
    print("a")
    print(a)
    print("choices")
    print(choices)
    print(f"mode = {mode}")
    _choices = list(choices)
    n = len(_choices)
    # broadcast and promote types as necessary
    choices_checked = False
    while not choices_checked:
        choices_checked = True
        for i, choice in enumerate(_choices):
            if a.shape != choice.shape:
                a, _choices[i] = numpy_style_broadcast(a, choice)
                choices_checked = False
            if _choices[0].dtype != choice.dtype:
                _choices[0], _choices[i] = ivy.promote_types_of_inputs(_choices[0],
                                                                       _choices[i])
                choices_checked = False
    # create composite array
    c = ivy.empty_like(_choices[0])
    if mode == 'raise':
        assert ivy.max(a) < n and ivy.min(a) >= 0, "choose:Invalid entry in index array"
    elif mode == 'clip':
        a = a.clip(0, n - 1)
    elif mode == 'wrap':
        a = ivy.abs(ivy.fmod(a, n))
    else:
        raise ValueError("Invalid mode")
    for index in ivy.ndindex(a.shape):
        c[index]= _choices[int(a[index])][index]

    if out is not None:
        if out.shape == c.shape:
            out[:] = c
            # ivy.inplace_update(out, c)
        else:
            raise ValueError(f"choose: output array shape should be {c.shape}")
    else:
        return c
