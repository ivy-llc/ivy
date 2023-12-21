# global
import ivy

# local
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_ivy_arrays_and_back
def sinc(x):
    if ivy.get_num_dims(x) == 0:
        x = ivy.astype(x, ivy.float64)
    return ivy.sinc(x)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def unwrap(p, discont=None, axis=-1, *, period=2 * ivy.pi):
    p = ivy.asarray(p)
    nd = p.ndim
    dd = ivy.diff(p, axis=axis)
    if discont is None:
        discont = period / 2
    slice1 = [slice(None, None)] * nd  # full slices
    slice1[axis] = ivy.slice(1, None)
    slice1 = ivy.tuple(slice1)
    dtype = ivy.result_type(dd, period)
    if ivy.issubdtype(dtype, ivy.integer):
        interval_high, rem = ivy.divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = ivy.mod(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        ivy.copyto(ddmod, interval_high, where=(ddmod == interval_low) & (dd > 0))
    ph_correct = ddmod - dd
    ivy.copyto(ph_correct, 0, where=ivy.abs(dd) < discont)
    up = ivy.array(p, copy=True, dtype=dtype)
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up
