# global
import functools
from typing import Callable

# local
import ivy
import ivy.functional.frontends.torch as torch_frontend


def _from_ivy_array_to_torch_frontend_tensor(x, nested=False, include_derived=None):
    if nested:
        return ivy.nested_map(
            x, _from_ivy_array_to_torch_frontend_tensor, include_derived, shallow=False
        )
    elif isinstance(x, ivy.Array) or ivy.is_native_array(x):
        a = torch_frontend.Tensor(x, _init_overload=True)
        return a
    return x


def _to_ivy_array(x):
    # if x is a native array return it as an ivy array
    if isinstance(x, ivy.NativeArray):
        return ivy.array(x)

    # else if x is a frontend torch Tensor (or any frontend "Tensor" actually) return the wrapped ivy array # noqa: E501
    elif hasattr(x, "ivy_array"):
        return x.ivy_array

    # else just return x
    return x


def _store_grads(out_tensor, fn, *args, **kwargs):
    all_inputs = list(args) + list(kwargs.values())
    func_inputs = []
    inputs_req_grad = False
    for input in all_inputs:
        if isinstance(input, torch_frontend.Tensor) and input.requires_grad:
            func_inputs.append(input)
            inputs_req_grad = True

    if not inputs_req_grad:
        return out_tensor, inputs_req_grad

    out_tensor.func_inputs = func_inputs

    # Compute gradients
    # TODO: edit when ivy.jac is fixed to take args & kwargs
    jac_fn = to_ivy_arrays_and_back(ivy.jac(outputs_to_native_arrays(fn)))
    jacs = jac_fn(*args)
    out_tensor.grads = [jacs]

    return out_tensor, inputs_req_grad


def handle_gradients(fn: Callable) -> Callable:
    """Store gradients and fn inputs if args requires_grad = True"""

    @functools.wraps(fn)
    def _handle_gradients(*args, **kwargs):
        ret = fn(*args, **kwargs)

        if isinstance(ret, tuple):
            new_ret = []
            for out_tensor in ret:
                res, inputs_req_grad = _store_grads(out_tensor, fn, *args, **kwargs)
                res.requires_grad = inputs_req_grad
                new_ret.append(res)
            new_ret = tuple(new_ret)
        else:
            new_ret, inputs_req_grad = _store_grads(ret, fn, *args, **kwargs)
            new_ret.requires_grad = inputs_req_grad

        if "requires_grad" in kwargs:
            new_ret.requires_grad = kwargs["requires_grad"]

        return new_ret

    return _handle_gradients


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_arrays_torch(*args, **kwargs):
        """
        Convert `Tensor` into `ivy.Array` instances.

        Convert all `Tensor` instances in both the positional and
        keyword arguments into `ivy.Array` instances, and then calls the
        function with the updated arguments.
        """
        # Remove out argument if present in kwargs
        if "out" in kwargs and not ivy.nested_any(
            kwargs["out"], lambda x: isinstance(x, (torch_frontend.Tensor, type(None)))
        ):
            raise ivy.utils.exceptions.IvyException(
                "Out argument must be an ivy.frontends.torch.Tensor object"
            )
        # convert all input arrays to ivy.Array instances
        new_args = ivy.nested_map(
            args, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        new_kwargs = ivy.nested_map(
            kwargs, _to_ivy_array, include_derived={tuple: True}, shallow=False
        )
        return fn(*new_args, **new_kwargs)

    return _inputs_to_ivy_arrays_torch


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def outputs_to_frontend_arrays_torch(*args, **kwargs):
        """
        Convert `ivy.Array` into `Tensor` instances.

        Call the function, and then converts all `ivy.Array` instances
        returned by the function into `Tensor` instances.
        """
        # call unmodified function
        # ToDo: Remove this default dtype setting
        #  once frontend specific backend setting is added
        set_default_dtype = False
        if not ("dtype" in kwargs and ivy.exists(kwargs["dtype"])) and all(
            [not (ivy.is_array(i) or hasattr(i, "ivy_array")) for i in args]
        ):
            if ivy.current_backend_str() == "jax":
                import jax

                jax.config.update("jax_enable_x64", True)
            ivy.set_default_int_dtype("int64")
            ivy.set_default_float_dtype(torch_frontend.get_default_dtype())
            set_default_dtype = True
        try:
            ret = fn(*args, **kwargs)
        finally:
            if set_default_dtype:
                ivy.unset_default_int_dtype()
                ivy.unset_default_float_dtype()
        # convert all arrays in the return to `torch_frontend.Tensor` instances
        ret = _from_ivy_array_to_torch_frontend_tensor(
            ret, nested=True, include_derived={tuple: True}
        )
        if "inplace" in kwargs and kwargs["inplace"]:
            first_array = ivy.func_wrapper._get_first_array(
                *args, array_fn=lambda x: hasattr(x, "ivy_array"), **kwargs
            )
            ivy.inplace_update(first_array, ret.ivy_array)
            return first_array
        else:
            return ret

    return outputs_to_frontend_arrays_torch


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Wrap `fn` so it receives and returns `ivy.Array` instances.

    Wrap `fn` so that input arrays are all converted to `ivy.Array`
    instances and return arrays are all converted to `Tensor` instances.
    """
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))


def outputs_to_native_arrays(fn: Callable):
    @functools.wraps(fn)
    def outputs_to_native_arrays_torch(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if isinstance(ret, torch_frontend.Tensor):
            ret = ret.ivy_array.data
        return ret

    return outputs_to_native_arrays_torch
