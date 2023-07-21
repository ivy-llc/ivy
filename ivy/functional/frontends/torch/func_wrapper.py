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


def _does_input_req_grad(inputs):
    def fn(x):
        return isinstance(x, torch_frontend.Tensor) and x.requires_grad

    return ivy.nested_any(inputs, fn)


def _remove_non_tensors(xs):
    idx_to_prune = ivy.nested_argwhere(
        xs,
        lambda x: not isinstance(x, torch_frontend.Tensor),
        to_ignore=torch_frontend.Tensor,
    )
    xs_pruned = ivy.copy_nest(xs, to_mutable=True)
    ivy.prune_nest_at_indices(xs_pruned, idx_to_prune)
    return xs_pruned, idx_to_prune


def _add_non_tensors(xs_torch, xs, prune_idxs):
    for idx in prune_idxs:
        to_add = ivy.index_nest(xs, idx)
        ivy.insert_into_nest_at_index(xs_torch, idx, to_add)
    return xs_torch


def _store_data(out_tensors, fn, xs):
    xs_pruned, prune_idxs = _remove_non_tensors(xs)

    def fn_wrapped(xs_pruned):
        # To frontend
        xs_torch = ivy.nested_map(
            xs_pruned,
            lambda x: torch_frontend.Tensor(x, _init_overload=True),
            shallow=False,
        )
        xs_torch_all = _add_non_tensors(xs_torch, xs, prune_idxs)

        ret = fn(*xs_torch_all[0], **xs_torch_all[1])

        # To ivy
        ivy_ret = ivy.nested_map(
            ret,
            _to_ivy_array,
            to_ignore=torch_frontend.Tensor,
            shallow=False,
            include_derived={tuple: True},
        )
        return ivy_ret

    jac_fn = to_ivy_arrays_and_back(ivy.jac(fn_wrapped))
    if isinstance(out_tensors, torch_frontend.Tensor):
        out_tensors.func_inputs = xs_pruned
        out_tensors.jac_fn = jac_fn
        out_tensors.requires_grad = True
    else:
        idxs = ivy.all_nested_indices(out_tensors)
        for idx in idxs:
            o = ivy.index_nest(out_tensors, idx)
            o.func_inputs = xs_pruned
            o.jac_fn = jac_fn
            o.requires_grad = True
            o.out_idx = idx

    return out_tensors


def handle_gradients(fn: Callable) -> Callable:
    """Store gradients and fn inputs."""

    @functools.wraps(fn)
    def _handle_gradients(*args, **kwargs):
        ret = fn(*args, **kwargs)

        xs = [args, kwargs]
        inputs_req_grad = _does_input_req_grad(xs)

        if inputs_req_grad:
            ret = _store_data(ret, fn, xs)
        if "requires_grad" in kwargs:
            ret.requires_grad = kwargs["requires_grad"]

        return ret

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
        array_fn = lambda x: ivy.is_array(x) or hasattr(x, "ivy_array")
        if "inplace" in kwargs and kwargs["inplace"]:
            first_array = ivy.func_wrapper._get_first_array(
                *args, array_fn=array_fn, **kwargs
            )
            # ivy.inplace_update with ensure_in_backend=True fails in jax and tf
            # so update ._data directly
            if ivy.is_array(first_array):
                first_array._data = ret.ivy_array._data
            else:
                first_array.ivy_array._data = ret.ivy_array._data
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


numpy_compatible_args = {
    "axis": "dim",
    "keepdims": "keepdim",
    "x": "input",
    "a": "input",
    "x1": "input",
    "x2": "other",
}


# noqa: F811
def numpy_to_torch_style_args(func):  # noqa
    """Convert argument names from NumPy style to PyTorch style."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_kwargs = {
            numpy_compatible_args.get(key, key): value for key, value in kwargs.items()
        }
        return func(*args, **new_kwargs)

    return wrapper
