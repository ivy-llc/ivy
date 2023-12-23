# global
import inspect
from typing import Callable, Dict, Optional
import functools

# local
import ivy
import ivy.functional.frontends.tensorflow as frontend
import ivy.functional.frontends.numpy as np_frontend


# --- Helpers --- #
# --------------- #


def _ivy_array_to_tensorflow(x):
    if isinstance(x, ivy.Array) or ivy.is_native_array(x):
        return frontend.EagerTensor(x)
    return x


def _native_to_ivy_array(x):
    if isinstance(x, ivy.NativeArray):
        return ivy.array(x)
    return x


def _tf_frontend_array_to_ivy(x):
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    return x


def _to_ivy_array(x):
    return _tf_frontend_array_to_ivy(_native_to_ivy_array(x))


# update kwargs dictionary keys helper
def _update_kwarg_keys(kwargs: Dict, to_update: Dict) -> Dict:
    """Update the key-word only arguments dictionary.

    Parameters
    ----------
    kwargs
        A dictionary containing key-word only arguments to be updated.

    to_update
        The dictionary containing keys to update from raw_ops function the mapping
        is raw_ops argument name against corresponding tf_frontend argument name.

    Returns
    -------
    ret
        An updated dictionary with new keyword mapping
    """
    new_kwargs = {}
    for key, value in kwargs.items():
        if to_update.__contains__(key):
            new_kwargs.update({to_update[key]: value})
        else:
            new_kwargs.update({key: value})
    return new_kwargs


# --- Main --- #
# ------------ #


def handle_tf_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_tf_dtype(*args, dtype=None, **kwargs):
        if len(args) > (dtype_pos + 1):
            dtype = args[dtype_pos]
            kwargs = {
                **dict(
                    zip(
                        list(inspect.signature(fn).parameters.keys())[
                            dtype_pos + 1 : len(args)
                        ],
                        args[dtype_pos + 1 :],
                    )
                ),
                **kwargs,
            }
            args = args[:dtype_pos]
        elif len(args) == (dtype_pos + 1):
            dtype = args[dtype_pos]
            args = args[:-1]
        if dtype is not None:
            dtype = to_ivy_dtype(dtype)
            return fn(*args, dtype=dtype, **kwargs)
        return fn(*args, **kwargs)

    dtype_pos = list(inspect.signature(fn).parameters).index("dtype")
    _handle_tf_dtype.handle_tf_dtype = True
    return _handle_tf_dtype


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_arrays_tf(*args, **kwargs):
        """Convert all `TensorFlow.Tensor` instances in both the positional and
        keyword arguments into `ivy.Array` instances, and then call the
        function with the updated arguments.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy arrays passed in the arguments.
        """
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True

        # convert all arrays in the inputs to ivy.Array instances
        ivy_args = ivy.nested_map(
            _to_ivy_array, args, include_derived=True, shallow=False
        )
        ivy_kwargs = ivy.nested_map(
            _to_ivy_array, kwargs, include_derived=True, shallow=False
        )
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)

    _inputs_to_ivy_arrays_tf.inputs_to_ivy_arrays_tf = True
    return _inputs_to_ivy_arrays_tf


def map_raw_ops_alias(
    alias: callable, kwargs_to_update: Optional[Dict] = None
) -> callable:
    """Map the raw_ops function with its respective frontend alias function, as
    the implementations of raw_ops is way similar to that of frontend
    functions, except that only arguments are passed as key-word only in
    raw_ops functions.

    Parameters
    ----------
    alias:
        The frontend function that is being referenced to as an alias to the
        current raw_ops function.
    kwargs_to_update:
        A dictionary containing key-word args to update to conform with a given
        raw_ops function

    Returns
    -------
    ret
        A decorated tf_frontend function to alias a given raw_ops function.
        Only accepting key-word only arguments.
    """

    def _wrap_raw_ops_alias(fn: callable, kw_update: Dict) -> callable:
        # removing decorators from frontend function
        fn = inspect.unwrap(fn)

        # changing all the params to keyword-only
        sig = inspect.signature(fn)
        new_params = []
        kw_update_rev = (
            {value: key for key, value in kw_update.items()} if kw_update else {}
        )
        for param in sig.parameters.values():
            # updating the name of the parameter
            name = (
                kw_update_rev[param.name]
                if kw_update and kw_update_rev.__contains__(param.name)
                else param.name
            )
            new_params.append(param.replace(name=name, kind=param.KEYWORD_ONLY))
        new_signature = sig.replace(parameters=new_params)

        def _wraped_fn(**kwargs):
            # update kwargs dictionary keys
            if kw_update:
                kwargs = _update_kwarg_keys(kwargs, kw_update)
            return fn(**kwargs)

        _wraped_fn.__signature__ = new_signature
        return _wraped_fn

    _wrap_raw_ops_alias.wrap_raw_ops_alias = True
    return _wrap_raw_ops_alias(alias, kwargs_to_update)


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_arrays_tf(*args, **kwargs):
        """Call the function, and then convert all `tensorflow.Tensor`
        instances in the function return into `ivy.Array` instances.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy arrays as tensorflow.Tensor arrays.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)

        # convert all arrays in the return to `frontend.Tensorflow.tensor` instances
        return ivy.nested_map(
            _ivy_array_to_tensorflow, ret, include_derived={"tuple": True}
        )

    _outputs_to_frontend_arrays_tf.outputs_to_frontend_arrays_tf = True
    return _outputs_to_frontend_arrays_tf


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    return outputs_to_frontend_arrays(inputs_to_ivy_arrays(fn))


def to_ivy_dtype(dtype):
    if not dtype or isinstance(dtype, str):
        return dtype
    if isinstance(dtype, np_frontend.dtype):
        return dtype.ivy_dtype
    return frontend.as_dtype(dtype).ivy_dtype
