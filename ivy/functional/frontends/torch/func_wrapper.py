# global
import functools
from typing import Callable

# local
import ivy
import ivy.functional.frontends.torch as torch_frontend


class AccumulateGrad:
    def __init__(self) -> None:
        self.next_functions = ()
        self.__name__ = "AccumulateGrad"

    def __repr__(self):
        return self.__name__

    def __eq__(self, __value: object) -> bool:
        return self.__name__ == __value

    def __call__(self, grads):
        for i in range(self.__self__.ndim):
            grads = grads.sum(-1)
        self.__self__._grads = grads
        return None


class GradFn:
    def __init__(self, fn, inputs) -> None:
        self._inputs = []
        self._fns = []
        self.next_functions = []
        assert len(inputs) <= 2
        if len(inputs) == 1 and isinstance(inputs[0], torch_frontend.Tensor):
            self._inputs.append(inputs[0].detach())
            d_fn = lambda x: fn(x)
            self._fns.append(to_ivy_arrays_and_back(ivy.jac(d_fn)))
            if inputs[0].grad_fn is not None:
                self.next_functions.append(inputs[0].grad_fn)
            elif inputs[0].requires_grad and inputs[0].is_leaf:
                acc_grad = AccumulateGrad()
                acc_grad.__self__ = inputs[0]
                self.next_functions.append(acc_grad)
        elif len(inputs) == 2:
            if isinstance(inputs[0], torch_frontend.Tensor):
                self._inputs.append(inputs[0].detach())
                d_fn = lambda x: fn(x, inputs[1])
                self._fns.append(to_ivy_arrays_and_back(ivy.jac(d_fn)))
                if inputs[0].grad_fn is not None:
                    self.next_functions.append(inputs[0].grad_fn)
                elif inputs[0].requires_grad and inputs[0].is_leaf:
                    acc_grad = AccumulateGrad()
                    acc_grad.__self__ = inputs[0]
                    self.next_functions.append(acc_grad)
            if isinstance(inputs[1], torch_frontend.Tensor):
                self._inputs.append(inputs[1].detach())
                d_fn = lambda x: fn(inputs[0], x)
                self._fns.append(to_ivy_arrays_and_back(ivy.jac(d_fn)))
                if inputs[1].grad_fn is not None:
                    self.next_functions.append(inputs[1].grad_fn)
                elif inputs[1].requires_grad and inputs[1].is_leaf:
                    acc_grad = AccumulateGrad()
                    acc_grad.__self__ = inputs[1]
                    self.next_functions.append(acc_grad)
        self.__name__ = fn.__name__.capitalize() + "Backward"

    def __call__(self, prev_grads):
        return [
            jac_fn(input_tensor) * prev_grads
            for input_tensor, jac_fn in zip(self._inputs, self._fns)
        ]

    def __repr__(self):
        return self.__name__

    def __eq__(self, __value: object) -> bool:
        return self.__name__ == __value


def _from_ivy_array_to_torch_frontend_tensor(
    x, nested=False, include_derived=None, requires_grad=False
):
    if nested:
        return ivy.nested_map(
            x,
            functools.partial(
                _from_ivy_array_to_torch_frontend_tensor, requires_grad=requires_grad
            ),
            include_derived,
            shallow=False,
        )
    elif isinstance(x, ivy.Array) or ivy.is_native_array(x):
        a = torch_frontend.Tensor(x, _init_overload=True, requires_grad=requires_grad)
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


def _store_data(out_tensors, fn, xs):
    if isinstance(out_tensors, torch_frontend.Tensor):
        out_tensors.func_inputs = xs
        out_tensors._func = torch_frontend.__dict__[fn.__name__]
        out_tensors.requires_grad = True
    else:
        idxs = ivy.all_nested_indices(out_tensors)
        for idx in idxs:
            o = ivy.index_nest(out_tensors, idx)
            if isinstance(o, torch_frontend.Tensor):
                o.func_inputs = xs
                o._func = torch_frontend.__dict__[fn.__name__]
                o.requires_grad = True
                o.out_idx = idx

    return out_tensors


def handle_gradients(fn: Callable) -> Callable:
    """Store fn and inputs."""

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
        """Convert all `Tensor` instances in both the positional and keyword arguments
        into `ivy.Array` instances, and then call the function with the updated
        arguments."""
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
        """Call the function, and then convert all `ivy.Array` instances returned by the
        function into `Tensor` instances."""
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
            ret,
            nested=True,
            include_derived={tuple: True},
            requires_grad=kwargs.get(
                "requires_grad",
                any(
                    [
                        isinstance(i, torch_frontend.Tensor) and i.requires_grad
                        for i in args
                    ]
                ),
            ),
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
            ret = first_array

        # logic for setting is_leaf
        if ret is not None and isinstance(ret, torch_frontend.Tensor):
            if fn.__name__ in dir(torch_frontend.creation_ops):
                ret.is_leaf = True
            elif all(
                [
                    not isinstance(i, torch_frontend.Tensor)
                    or (not i.requires_grad and not i.grad_fn)
                    for i in args
                ]
            ):
                ret.is_leaf = True
            else:
                ret.is_leaf = False
        # set grad_fn
        if any(
            [isinstance(i, torch_frontend.Tensor) and i.requires_grad for i in args]
        ):
            # ToDo: Implement for unbind
            grad_fn = GradFn(fn, args)
            grad_fn.__self__ = ret
            ret.grad_fn = grad_fn

        return ret

    return outputs_to_frontend_arrays_torch


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """Wrap `fn` so that input arrays are all converted to `ivy.Array` instances and
    return arrays are all converted to `Tensor` instances."""
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))


def outputs_to_native_arrays(fn: Callable):
    @functools.wraps(fn)
    def outputs_to_native_arrays_torch(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if isinstance(ret, torch_frontend.Tensor):
            ret = ret.ivy_array.data
        return ret

    return outputs_to_native_arrays_torch


def to_ivy_shape(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def to_ivy_shape_torch(*args, **kwargs):
        new_kwargs = {
            key: (
                value.ivy_shape
                if key in ["shape", "size"]
                and isinstance(value, ivy.functional.frontends.torch.Size)
                else value
            )
            for key, value in kwargs.items()
        }
        # if any of the args are instance of torch_frontend.Size,
        # convert them to ivy.Shape.
        new_args = ivy.nested_map(
            args,
            lambda x: (
                x.ivy_shape if isinstance(x, ivy.functional.frontends.torch.Size) else x
            ),
            shallow=False,
        )
        return fn(*new_args, **new_kwargs)

    return to_ivy_shape_torch


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
