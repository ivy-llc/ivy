import ivy
from ivy.functional.frontends.jax.devicearray import DeviceArray
from ivy.functional.frontends.numpy.func_wrapper import handle_numpy_dtype


@handle_numpy_dtype
def array(object, dtype=None, copy=True, order="K", ndmin=0):
    # TODO must ensure the array is created on default device.
    if order is not None and order != "K":
        raise ivy.exceptions.IvyNotImplementedException(
            "Only implemented for order='K'"
        )
    ret = ivy.array(object, dtype=dtype)
    if ivy.get_num_dims(ret) < ndmin:
        ret = ivy.expand_dims(ret, axis=list(range(ndmin - ivy.get_num_dims(ret))))
    return DeviceArray(ret)
