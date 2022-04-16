"""
Collection of helpers for ivy unit tests
"""

# global
import numpy as np
try:
    import jax.numpy as _jnp
except ImportError:
    _jnp = None
try:
    import tensorflow as _tf
    _tf_version = float('.'.join(_tf.__version__.split('.')[0:2]))
    if _tf_version >= 2.3:
        # noinspection PyPep8Naming,PyUnresolvedReferences
        from tensorflow.python.types.core import Tensor as tensor_type
    else:
        # noinspection PyPep8Naming
        # noinspection PyProtectedMember,PyUnresolvedReferences
        from tensorflow.python.framework.tensor_like import _TensorLike as tensor_type
    physical_devices = _tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        _tf.config.experimental.set_memory_growth(device, True)
except ImportError:
    _tf = None
try:
    import torch as _torch
except ImportError:
    _torch = None
try:
    import mxnet as _mx
    import mxnet.ndarray as _mx_nd
except ImportError:
    _mx = None
    _mx_nd = None
from hypothesis import strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np


def get_ivy_numpy():
    try:
        import ivy.functional.backends.numpy
    except ImportError:
        return None
    return ivy.functional.backends.numpy


def get_ivy_jax():
    try:
        import ivy.functional.backends.jax
    except ImportError:
        return None
    return ivy.functional.backends.jax


def get_ivy_tensorflow():
    try:
        import ivy.functional.backends.tensorflow
    except ImportError:
        return None
    return ivy.functional.backends.tensorflow


def get_ivy_torch():
    try:
        import ivy.functional.backends.torch
    except ImportError:
        return None
    return ivy.functional.backends.torch


def get_ivy_mxnet():
    try:
        import ivy.functional.backends.mxnet
    except ImportError:
        return None
    return ivy.functional.backends.mxnet


_ivy_fws_dict = {'numpy': lambda: get_ivy_numpy(),
                 'jax': lambda: get_ivy_jax(),
                 'tensorflow': lambda: get_ivy_tensorflow(),
                 'tensorflow_graph': lambda: get_ivy_tensorflow(),
                 'torch': lambda: get_ivy_torch(),
                 'mxnet': lambda: get_ivy_mxnet()}

_iterable_types = [list, tuple, dict]
_excluded = []


def _convert_vars(vars_in, from_type, to_type_callable=None, keep_other=True, to_type=None):
    new_vars = list()
    for var in vars_in:
        if type(var) in _iterable_types:
            return_val = _convert_vars(var, from_type, to_type_callable)
            new_vars.append(return_val)
        elif isinstance(var, from_type):
            if isinstance(var, np.ndarray):
                if var.dtype == np.float64:
                    var = var.astype(np.float32)
                if bool(sum([stride < 0 for stride in var.strides])):
                    var = var.copy()
            if to_type_callable:
                new_vars.append(to_type_callable(var))
            else:
                raise Exception('Invalid. A conversion callable is required.')
        elif to_type is not None and isinstance(var, to_type):
            new_vars.append(var)
        elif keep_other:
            new_vars.append(var)

    return new_vars


def np_call(func, *args, **kwargs):
    ret = func(*args, **kwargs)
    if isinstance(ret, (list, tuple)):
        return ivy.to_native(ret, nested=True)
    return ivy.to_numpy(ret)


def jnp_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _jnp.asarray)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, _jnp.asarray)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (_jnp.ndarray, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (_jnp.ndarray, ivy.Array), ivy.to_numpy)[0]


def tf_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _tf.convert_to_tensor)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, _tf.convert_to_tensor)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (tensor_type, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (tensor_type, ivy.Array), ivy.to_numpy)[0]


def tf_graph_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _tf.convert_to_tensor)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, _tf.convert_to_tensor)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))

    @_tf.function
    def tf_func(*local_args, **local_kwargs):
        return func(*local_args, **local_kwargs)

    output = tf_func(*new_args, **new_kwargs)

    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (tensor_type, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (tensor_type, ivy.Array), ivy.to_numpy)[0]


def torch_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _torch.from_numpy)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, _torch.from_numpy)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (_torch.Tensor, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (_torch.Tensor, ivy.Array), ivy.to_numpy)[0]


def mx_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, _mx_nd.array)
    new_kw_items = _convert_vars(kwargs.values(), np.ndarray, _mx_nd.array)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_items))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (_mx_nd.ndarray.NDArray, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (_mx_nd.ndarray.NDArray, ivy.Array), ivy.to_numpy)[0]


_calls = [np_call, jnp_call, tf_call, tf_graph_call, torch_call, mx_call]


def assert_compilable(fn):
    try:
        ivy.compile(fn)
    except Exception as e:
        raise e


def assert_docstring_examples_run(fn):
    fn_name = fn.__name__
    docstring = ivy.framework_handler.ivy_original_dict[fn_name].__doc__
    if docstring is None:
        return True
    executable_lines = [line.split('>>>')[1][1:] for line in docstring.split('\n') if '>>>' in line]
    for line in executable_lines:
        exec(line)
    return True


def var_fn(a, b=None, c=None):
    return ivy.variable(ivy.array(a, b, c))


def exclude(exclusion_list):
    global _excluded
    _excluded += list(set(exclusion_list) - set(_excluded))


def frameworks():
    return list(set([ivy_fw() for fw_str, ivy_fw in _ivy_fws_dict.items()
                     if ivy_fw() is not None and fw_str not in _excluded]))


def calls():
    return [call for (fw_str, ivy_fw), call in zip(_ivy_fws_dict.items(), _calls)
            if ivy_fw() is not None and fw_str not in _excluded]


def f_n_calls():
    return [(ivy_fw(), call) for (fw_str, ivy_fw), call in zip(_ivy_fws_dict.items(), _calls)
            if ivy_fw() is not None and fw_str not in _excluded]


def sample(iterable):
    return st.builds(lambda i: iterable[i], st.integers(0, len(iterable) - 1))


def assert_all_close(x, y):
    assert np.allclose(np.nan_to_num(x), np.nan_to_num(y))


def kwargs_to_args_n_kwargs(positional_ratio, kwargs):
    num_args_n_kwargs = len(kwargs)
    num_args = int(round(positional_ratio * num_args_n_kwargs))
    args = [v for v in list(kwargs.values())[:num_args]]
    kwargs = {k: kwargs[k] for k in list(kwargs.keys())[num_args:]}
    return args, kwargs


def test_array_function(dtype, as_variable, with_out, native_array, positional_ratio, fw, fn_name, **kwargs):
    args, kwargs = kwargs_to_args_n_kwargs(positional_ratio, kwargs)
    if dtype in ivy.invalid_dtype_strs:
        return  # invalid dtype
    args = ivy.nested_map(args, lambda x: ivy.array(x, dtype=dtype) if isinstance(x, np.ndarray) else x)
    kwargs = ivy.nested_map(kwargs, lambda x: ivy.array(x, dtype=dtype) if isinstance(x, np.ndarray) else x)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            return  # only floating point variables are supported
        if with_out:
            return  # variables do not support out argument
        args = ivy.nested_map(args, lambda x: ivy.variable(x) if ivy.is_array(x) else x)
        kwargs = ivy.nested_map(kwargs, lambda x: ivy.variable(x) if ivy.is_array(x) else x)
    if native_array:
        args, kwargs = ivy.args_to_native(*args, **kwargs)
    ret = ivy.__dict__[fn_name](*args, **kwargs)
    out = ret
    if with_out:
        assert not isinstance(ret, tuple)
        assert ivy.is_array(ret)
        if as_variable:
            out = ivy.variable(out)
        if native_array:
            out = out.data
        ret = ivy.__dict__[fn_name](*args, **kwargs, out=out)
        if not native_array:
            assert ret is out
        if fw in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)
    # value test
    if dtype == 'bfloat16':
        return  # bfloat16 is not supported by numpy
    ret_idxs = ivy.nested_indices_where(ret, ivy.is_array)
    ret_flat = ivy.multi_index_nest(ret, ret_idxs)
    ret_np_flat = [ivy.to_numpy(x) for x in ret_flat]
    args_np = ivy.nested_map(args, lambda x: ivy.to_numpy(x) if ivy.is_array(x) else x)
    kwargs_np = ivy.nested_map(kwargs, lambda x: ivy.to_numpy(x) if ivy.is_array(x) else x)
    ret_from_np = ivy_np.__dict__[fn_name](*args_np, **kwargs_np)
    ret_from_np_flat = ivy.multi_index_nest(ret_from_np, ret_idxs)
    for ret_np, ret_from_np in zip(ret_np_flat, ret_from_np_flat):
        assert_all_close(ret_np, ret_from_np)
