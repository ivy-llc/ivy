# global
import copy
from typing import Union, List
import numpy as np
import types
import importlib
import inspect

try:
    import tensorflow as tf
except ImportError:
    tf = types.SimpleNamespace()
    tf.TensorShape = None

# local
import ivy
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy.functional.ivy.gradients import _variable
from ivy_tests.test_ivy.test_frontends import NativeClass
from ivy_tests.test_ivy.helpers.structs import FrontendMethodData
from ivy.functional.frontends.torch.tensor import Tensor as torch_tensor
from ivy.functional.frontends.tensorflow.tensor import EagerTensor as tf_tensor
from ivy.functional.frontends.jax.devicearray import DeviceArray
from ivy.functional.frontends.numpy.ndarray.ndarray import ndarray
from .assertions import (
    value_test,
    check_unsupported_dtype,
)


def empty_func(*args, **kwargs):
    return None


try:
    from ivy.functional.backends.jax.general import (
        is_native_array as is_jax_native_array,
    )
except ImportError:
    is_jax_native_array = empty_func

try:
    from ivy.functional.backends.numpy.general import (
        is_native_array as is_numpy_native_array,
    )
except ImportError:
    is_numpy_native_array = empty_func

try:
    from ivy.functional.backends.tensorflow.general import (
        is_native_array as is_tensorflow_native_array,
    )
except ImportError:
    is_tensorflow_native_array = empty_func

try:
    from ivy.functional.backends.torch.general import (
        is_native_array as is_torch_native_array,
    )
except ImportError:
    is_torch_native_array = empty_func


# ToDo, this is temporary until unsupported_dtype is embedded
# into helpers.get_dtypes
def _assert_dtypes_are_valid(input_dtypes: Union[List[ivy.Dtype], List[str]]):
    for dtype in input_dtypes:
        if dtype not in ivy.valid_dtypes + ivy.valid_complex_dtypes:
            raise Exception(f"{dtype} is not a valid data type.")


# Function testing


def test_function(
    *,
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: List[bool],
    with_out: bool,
    num_positional_args: int,
    native_array_flags: List[bool],
    container_flags: List[bool],
    instance_method: bool,
    fw: str,
    fn_name: str,
    rtol_: float = None,
    atol_: float = 1e-06,
    test_values: bool = True,
    test_gradients: bool = False,
    xs_grad_idxs=None,
    ret_grad_idxs=None,
    ground_truth_backend: str,
    on_device: str = "cpu",
    return_flat_np_arrays: bool = False,
    **all_as_kwargs_np,
):
    """Tests a function that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated
        as a variable.
    with_out
        if True, the function is also tested with the optional out argument.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    native_array_flags
        dictates whether the corresponding input argument should be treated
        as a native array.
    container_flags
        dictates whether the corresponding input argument should be treated
         as an ivy Container.
    instance_method
        if True, the function is run as an instance method of the first
         argument (should be an ivy Array or Container).
    fw
        current backend (framework).
    fn_name
        name of the function to test.
    rtol_
        relative tolerance value.
    atol_
        absolute tolerance value.
    test_values
        if True, test for the correctness of the resulting values.
    test_gradients
        if True, test for the correctness of gradients.
    ground_truth_backend
        Ground Truth Backend to compare the result-values.
    on_device
        The device on which to create arrays
    return_flat_np_arrays
        If test_values is False, this flag dictates whether the original returns are
        returned, or whether the flattened numpy arrays are returned.
    all_as_kwargs_np
        input arguments to the function as keyword arguments.

    Returns
    -------
    ret
        optional, return value from the function
    ret_gt
        optional, return value from the Ground Truth function

    Examples
    --------
    >>> input_dtypes = 'float64'
    >>> as_variable_flags = False
    >>> with_out = False
    >>> num_positional_args = 0
    >>> native_array_flags = False
    >>> container_flags = False
    >>> instance_method = False
    >>> fw = "torch"
    >>> fn_name = "abs"
    >>> x = np.array([-1])
    >>> test_function(input_dtypes, as_variable_flags, with_out,\
                            num_positional_args, native_array_flags,\
                            container_flags, instance_method, fw, fn_name, x=x)

    >>> input_dtypes = ['float64', 'float32']
    >>> as_variable_flags = [False, True]
    >>> with_out = False
    >>> num_positional_args = 1
    >>> native_array_flags = [True, False]
    >>> container_flags = [False, False]
    >>> instance_method = False
    >>> fw = "numpy"
    >>> fn_name = "add"
    >>> x1 = np.array([1, 3, 4])
    >>> x2 = np.array([-3, 15, 24])
    >>> test_function(input_dtypes, as_variable_flags, with_out,\
                            num_positional_args, native_array_flags,\
                             container_flags, instance_method,\
                              fw, fn_name, x1=x1, x2=x2)
    """
    _assert_dtypes_are_valid(input_dtypes)
    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
    )

    # extract all arrays from the arguments and keyword arguments
    arg_np_vals, args_idxs, c_arg_vals = _get_nested_np_arrays(args_np)
    kwarg_np_vals, kwargs_idxs, c_kwarg_vals = _get_nested_np_arrays(kwargs_np)

    # make all lists equal in length
    num_arrays = c_arg_vals + c_kwarg_vals
    if len(input_dtypes) < num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(num_arrays)]
    if len(as_variable_flags) < num_arrays:
        as_variable_flags = [as_variable_flags[0] for _ in range(num_arrays)]
    if len(native_array_flags) < num_arrays:
        native_array_flags = [native_array_flags[0] for _ in range(num_arrays)]
    if len(container_flags) < num_arrays:
        container_flags = [container_flags[0] for _ in range(num_arrays)]

    # update variable flags to be compatible with float dtype and with_out args
    as_variable_flags = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]

    # update instance_method flag to only be considered if the
    # first term is either an ivy.Array or ivy.Container
    instance_method = instance_method and (
        not native_array_flags[0] or container_flags[0]
    )

    fn = getattr(ivy, fn_name)
    if gradient_incompatible_function(fn=fn):
        return

    args, kwargs, num_arg_vals, args_idxs, kwargs_idxs = create_args_kwargs(
        args_np=args_np,
        arg_np_vals=arg_np_vals,
        args_idxs=args_idxs,
        kwargs_np=kwargs_np,
        kwarg_np_vals=kwarg_np_vals,
        kwargs_idxs=kwargs_idxs,
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable_flags,
        native_array_flags=native_array_flags,
        container_flags=container_flags,
    )

    # run either as an instance method or from the API directly
    instance = None
    if instance_method:
        is_instance = [
            (not native_flag) or container_flag
            for native_flag, container_flag in zip(native_array_flags, container_flags)
        ]
        arg_is_instance = is_instance[:num_arg_vals]
        kwarg_is_instance = is_instance[num_arg_vals:]
        if arg_is_instance and max(arg_is_instance):
            i = 0
            for i, a in enumerate(arg_is_instance):
                if a:
                    break
            instance_idx = args_idxs[i]
            instance = ivy.index_nest(args, instance_idx)
            args = ivy.copy_nest(args, to_mutable=False)
            ivy.prune_nest_at_index(args, instance_idx)
        else:
            i = 0
            for i, a in enumerate(kwarg_is_instance):
                if a:
                    break
            instance_idx = kwargs_idxs[i]
            instance = ivy.index_nest(kwargs, instance_idx)
            kwargs = ivy.copy_nest(kwargs, to_mutable=False)
            ivy.prune_nest_at_index(kwargs, instance_idx)
        ret, ret_np_flat = get_ret_and_flattened_np_array(
            instance.__getattribute__(fn_name), *args, **kwargs
        )
    else:
        ret, ret_np_flat = get_ret_and_flattened_np_array(
            ivy.__dict__[fn_name], *args, **kwargs
        )
    # assert idx of return if the idx of the out array provided
    if with_out:
        test_ret = ret
        if isinstance(ret, tuple):
            assert hasattr(ivy.__dict__[fn_name], "out_index")
            test_ret = ret[getattr(ivy.__dict__[fn_name], "out_index")]
        out = ivy.zeros_like(test_ret)
        if max(container_flags):
            assert ivy.is_ivy_container(test_ret)
        else:
            assert ivy.is_array(test_ret)
        if instance_method:
            ret, ret_np_flat = get_ret_and_flattened_np_array(
                instance.__getattribute__(fn_name), *args, **kwargs, out=out
            )
        else:
            ret, ret_np_flat = get_ret_and_flattened_np_array(
                ivy.__dict__[fn_name], *args, **kwargs, out=out
            )
        test_ret = ret
        if isinstance(ret, tuple):
            test_ret = ret[getattr(ivy.__dict__[fn_name], "out_index")]
        assert test_ret is out
        if not max(container_flags) and ivy.native_inplace_support:
            # these backends do not always support native inplace updates
            assert test_ret.data is out.data
    # compute the return with a Ground Truth backend
    ivy.set_backend(ground_truth_backend)
    try:
        fn = getattr(ivy, fn_name)
        args, kwargs, _, _, _ = create_args_kwargs(
            args_np=args_np,
            arg_np_vals=arg_np_vals,
            args_idxs=args_idxs,
            kwargs_np=kwargs_np,
            kwargs_idxs=kwargs_idxs,
            kwarg_np_vals=kwarg_np_vals,
            input_dtypes=input_dtypes,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
            container_flags=container_flags,
        )
        ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
            ivy.__dict__[fn_name], *args, **kwargs
        )
    except Exception as e:
        ivy.unset_backend()
        raise e
    hasattr_unsupported_gradients = hasattr(fn, "unsupported_gradients")
    if hasattr_unsupported_gradients:
        fw_list = fn.unsupported_gradients
    else:
        fw_list = None
    ivy.unset_backend()
    # gradient test
    fw = ivy.current_backend_str()
    if (
        test_gradients
        and not fw == "numpy"
        and not instance_method
        and "bool" not in input_dtypes
    ):
        if hasattr_unsupported_gradients and fw in fw_list:
            if ivy.nested_argwhere(
                all_as_kwargs_np,
                lambda x: x.dtype in fw_list[fw] if isinstance(x, np.ndarray) else None,
            ):
                pass
            else:
                gradient_test(
                    fn_name=fn_name,
                    all_as_kwargs_np=all_as_kwargs_np,
                    args_np=args_np,
                    kwargs_np=kwargs_np,
                    input_dtypes=input_dtypes,
                    as_variable_flags=as_variable_flags,
                    native_array_flags=native_array_flags,
                    container_flags=container_flags,
                    rtol_=rtol_,
                    atol_=atol_,
                    xs_grad_idxs=xs_grad_idxs,
                    ret_grad_idxs=ret_grad_idxs,
                    ground_truth_backend=ground_truth_backend,
                )

        else:
            gradient_test(
                fn_name=fn_name,
                all_as_kwargs_np=all_as_kwargs_np,
                args_np=args_np,
                kwargs_np=kwargs_np,
                input_dtypes=input_dtypes,
                as_variable_flags=as_variable_flags,
                native_array_flags=native_array_flags,
                container_flags=container_flags,
                rtol_=rtol_,
                atol_=atol_,
                xs_grad_idxs=xs_grad_idxs,
                ret_grad_idxs=ret_grad_idxs,
                ground_truth_backend=ground_truth_backend,
            )

    # assuming value test will be handled manually in the test function
    if not test_values:
        if return_flat_np_arrays:
            return ret_np_flat, ret_np_from_gt_flat
        return ret, ret_from_gt

    if isinstance(rtol_, dict):
        rtol_ = _get_framework_rtol(rtol_, fw)
    if isinstance(atol_, dict):
        atol_ = _get_framework_atol(atol_, fw)

    # value test
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=ret_np_from_gt_flat,
        rtol=rtol_,
        atol=atol_,
        ground_truth_backend=ground_truth_backend,
    )


def test_frontend_function(
    *,
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: List[bool],
    with_out: bool,
    with_inplace: bool = False,
    all_aliases: List[str] = None,
    num_positional_args: int,
    native_array_flags: List[bool],
    on_device="cpu",
    frontend: str,
    fn_tree: str,
    rtol: float = None,
    atol: float = 1e-06,
    test_values: bool = True,
    **all_as_kwargs_np,
):
    """Tests a frontend function for the current backend by comparing the result with
    the function in the associated framework.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated
        as an ivy Variable.
    with_out
        if True, the function is also tested for inplace update to an array
        passed to the optional out argument.
    with_inplace
        if True, the function is only tested with direct inplace update back to
        the inputted array and ignore the value of with_out.
    all_aliases
        a list of strings containing all aliases for that function
        in the current frontend with their full namespaces.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    native_array_flags
        dictates whether the corresponding input argument should be treated
        as a native array.
    frontend
        current frontend (framework).
    fn_tree
        Path to function in frontend framework namespace.
    rtol
        relative tolerance value.
    atol
        absolute tolerance value.
    test_values
        if True, test for the correctness of the resulting values.
    all_as_kwargs_np
        input arguments to the function as keyword arguments.

    Returns
    -------
    ret
        optional, return value from the function
    ret_np
        optional, return value from the Numpy function
    """
    assert (
        not with_out or not with_inplace
    ), "only one of with_out or with_inplace can be set as True"

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
    )

    # extract all arrays from the arguments and keyword arguments
    arg_np_vals, args_idxs, c_arg_vals = _get_nested_np_arrays(args_np)
    kwarg_np_vals, kwargs_idxs, c_kwarg_vals = _get_nested_np_arrays(kwargs_np)

    # make all lists equal in length
    num_arrays = c_arg_vals + c_kwarg_vals
    if len(input_dtypes) < num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(num_arrays)]
    if len(as_variable_flags) < num_arrays:
        as_variable_flags = [as_variable_flags[0] for _ in range(num_arrays)]
    if len(native_array_flags) < num_arrays:
        native_array_flags = [native_array_flags[0] for _ in range(num_arrays)]

    # update var flags to be compatible with float dtype and with_out args
    as_variable_flags = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]

    # frontend function
    # parse function name and frontend submodules (jax.lax, jax.numpy etc.)

    def _get_function(fn_tree):
        split_index = fn_tree.rfind(".")
        fn_mod, fn_name = fn_tree[:split_index], fn_tree[split_index + 1 :]
        function_module = importlib.import_module(fn_mod)
        function = function_module.__dict__[fn_name]
        return function, function_module, fn_name, fn_mod

    function, function_module, fn_name, frontend_submods = _get_function(
        fn_tree=fn_tree
    )

    # check for unsupported dtypes in backend framework
    def _test_backend_unsupported():
        args, kwargs, _, _, _ = create_args_kwargs(
            args_np=args_np,
            arg_np_vals=arg_np_vals,
            args_idxs=args_idxs,
            kwargs_np=kwargs_np,
            kwarg_np_vals=kwarg_np_vals,
            kwargs_idxs=kwargs_idxs,
            input_dtypes=input_dtypes,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
        )
        args_ivy, kwargs_ivy = ivy.args_to_ivy(
            *args, **kwargs
        )  # ToDo, probably redundant?
        return args, kwargs, args_ivy, kwargs_ivy

    args, kwargs, args_ivy, kwargs_ivy = _test_backend_unsupported()

    def _test_frontend_function(args, kwargs, args_ivy, kwargs_ivy):
        # frontend function
        frontend_fn = getattr(function_module, fn_name)
        # check and replace NativeClass object in arguments with ivy counterparts
        from ivy_tests.test_ivy.test_frontends.test_numpy import convnumpy

        convs = {"numpy": convnumpy}

        if "torch" in available_frameworks:
            from ivy_tests.test_ivy.test_frontends.test_torch import convtorch

            convs["torch"] = convtorch

        if "tensorflow" in available_frameworks:
            from ivy_tests.test_ivy.test_frontends.test_tensorflow import convtensor

            convs["tensorflow"] = convtensor

        if "jax" in available_frameworks:
            from ivy_tests.test_ivy.test_frontends.test_jax import convjax

            convs["jax"] = convjax

        if frontend in convs:
            conv = convs[frontend]
            args = ivy.nested_map(args, fn=conv, include_derived=True)
            kwargs = ivy.nested_map(kwargs, fn=conv, include_derived=True)

        # Make copy for arguments for functions that might use
        # inplace update by default
        copy_kwargs = copy.deepcopy(kwargs)
        copy_args = copy.deepcopy(args)
        # strip the decorator to get an Ivy array
        ret = get_frontend_ret(frontend_fn, *args_ivy, **kwargs_ivy)
        if with_out:
            if not inspect.isclass(ret):
                is_ret_tuple = issubclass(ret.__class__, tuple)
            else:
                is_ret_tuple = issubclass(ret, tuple)
            if is_ret_tuple:
                ret = ivy.nested_map(
                    ret,
                    lambda _x: ivy.array(_x) if not ivy.is_array(_x) else _x,
                    include_derived=True,
                )
            elif not ivy.is_array(ret):
                ret = ivy.array(ret)
            out = ret
            # pass return value to out argument
            # check if passed reference is correctly updated
            kwargs["out"] = out
            if is_ret_tuple:
                flatten_ret = flatten(ret=ret)
                flatten_out = flatten(ret=out)
                for ret_array, out_array in zip(flatten_ret, flatten_out):
                    if ivy.native_inplace_support:
                        assert ret_array.data is out_array.data
                    assert ret_array is out_array
            else:
                if ivy.native_inplace_support:
                    assert ret.data is out.data
                assert ret is out
        elif with_inplace:
            assert not isinstance(ret, tuple)
            assert ivy.is_array(ret)
            if "inplace" in list(inspect.signature(frontend_fn).parameters.keys()):
                # the function provides optional inplace update
                # set inplace update to be True and check
                # if returned reference is inputted reference
                # and if inputted reference's content is correctly updated
                copy_kwargs["inplace"] = True
                first_array = ivy.func_wrapper._get_first_array(
                    *copy_args, **copy_kwargs
                )
                ret_ = get_frontend_ret(frontend_fn, *copy_args, **copy_args)
                if ivy.native_inplace_support:
                    assert ret_.data is first_array.data
                assert first_array is ret_
            else:
                # the function provides inplace update by default
                # check if returned reference is inputted reference
                first_array = ivy.func_wrapper._get_first_array(*args, **kwargs)
                if ivy.native_inplace_support:
                    assert ret.data is first_array.data
                assert first_array is ret
                args, kwargs = copy_args, copy_kwargs

        # create NumPy args
        args_np = ivy.nested_map(
            args_ivy,
            lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
        )
        kwargs_np = ivy.nested_map(
            kwargs_ivy,
            lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
        )

        # temporarily set frontend framework as backend
        ivy.set_backend(frontend)
        try:
            # create frontend framework args
            args_frontend = ivy.nested_map(
                args_np,
                lambda x: ivy.native_array(x)
                if isinstance(x, np.ndarray)
                else ivy.as_native_dtype(x)
                if isinstance(x, ivy.Dtype)
                else x,
            )
            kwargs_frontend = ivy.nested_map(
                kwargs_np,
                lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
            )

            # change ivy dtypes to native dtypes
            if "dtype" in kwargs_frontend:
                kwargs_frontend["dtype"] = ivy.as_native_dtype(kwargs_frontend["dtype"])

            # change ivy device to native devices
            if "device" in kwargs_frontend:
                kwargs_frontend["device"] = ivy.as_native_dev(kwargs_frontend["device"])

            # check and replace the NativeClass objects in arguments
            # with true counterparts
            args_frontend = ivy.nested_map(
                args_frontend, fn=convtrue, include_derived=True, max_depth=10
            )
            kwargs_frontend = ivy.nested_map(
                kwargs_frontend, fn=convtrue, include_derived=True, max_depth=10
            )

            # compute the return via the frontend framework
            frontend_fw = importlib.import_module(fn_tree[25 : fn_tree.rfind(".")])
            frontend_ret = frontend_fw.__dict__[fn_name](
                *args_frontend, **kwargs_frontend
            )

            if ivy.isscalar(frontend_ret):
                frontend_ret_np_flat = [np.asarray(frontend_ret)]
            else:
                # tuplify the frontend return
                if not isinstance(frontend_ret, tuple):
                    frontend_ret = (frontend_ret,)
                frontend_ret_idxs = ivy.nested_argwhere(
                    frontend_ret, ivy.is_native_array
                )
                frontend_ret_flat = ivy.multi_index_nest(
                    frontend_ret, frontend_ret_idxs
                )
                frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
        except Exception as e:
            ivy.unset_backend()
            raise e
        # unset frontend framework from backend
        ivy.unset_backend()

        ret_np_flat = flatten_and_to_np(ret=ret)
        # assuming value test will be handled manually in the test function
        if not test_values:
            return ret, frontend_ret
        # value tests, iterating through each array in the flattened returns

        nonlocal rtol
        nonlocal atol

        if isinstance(rtol, dict):
            rtol = _get_framework_rtol(rtol, ivy.backend)
        if isinstance(atol, dict):
            atol = _get_framework_atol(atol, ivy.backend)

        value_test(
            ret_np_flat=ret_np_flat,
            ret_np_from_gt_flat=frontend_ret_np_flat,
            rtol=rtol,
            atol=atol,
            ground_truth_backend=frontend,
        )
        return ret, frontend_ret

    # Call the frontend testing function

    ret, frontend_ret = _test_frontend_function(args, kwargs, args_ivy, kwargs_ivy)

    # testing all alias functions
    if all_aliases is not None:
        # for each alias in aliases list
        for alias in all_aliases:
            function, function_module, fn_name, frontend_submods = _get_function(
                fn_tree=f"ivy.functional.frontends.{frontend}.{alias}"
            )

            # testing unsupported in that backend
            (
                args,
                kwargs,
                args_ivy,
                kwargs_ivy,
            ) = _test_backend_unsupported()

            # calling the testing function
            _test_frontend_function(args, kwargs, args_ivy, kwargs_ivy)
    if not test_values:
        return ret, frontend_ret


# Method testing


def gradient_test(
    *,
    fn_name,
    all_as_kwargs_np,
    args_np,
    kwargs_np,
    input_dtypes,
    as_variable_flags,
    native_array_flags,
    container_flags,
    rtol_: float = None,
    atol_: float = 1e-06,
    xs_grad_idxs=None,
    ret_grad_idxs=None,
    ground_truth_backend: str,
):
    def grad_fn(all_args):
        args, kwargs = all_args
        ret = ivy.__dict__[fn_name](*args, **kwargs)
        return ivy.nested_map(ret, ivy.mean, include_derived=True)

    # extract all arrays from the arguments and keyword arguments
    arg_np_vals, args_idxs, c_arg_vals = _get_nested_np_arrays(args_np)
    kwarg_np_vals, kwargs_idxs, c_kwarg_vals = _get_nested_np_arrays(kwargs_np)

    args, kwargs, _, args_idxs, kwargs_idxs = create_args_kwargs(
        args_np=args_np,
        arg_np_vals=arg_np_vals,
        args_idxs=args_idxs,
        kwargs_np=kwargs_np,
        kwarg_np_vals=kwarg_np_vals,
        kwargs_idxs=kwargs_idxs,
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable_flags,
        native_array_flags=native_array_flags,
        container_flags=container_flags,
    )
    _, grads = ivy.execute_with_gradients(
        grad_fn, [args, kwargs], xs_grad_idxs=xs_grad_idxs, ret_grad_idxs=ret_grad_idxs
    )
    grads_np_flat = flatten_and_to_np(ret=grads)

    # compute the return with a Ground Truth backend
    ivy.set_backend(ground_truth_backend)
    test_unsupported = check_unsupported_dtype(
        fn=ivy.__dict__[fn_name],
        input_dtypes=input_dtypes,
        all_as_kwargs_np=all_as_kwargs_np,
    )
    if test_unsupported:
        return
    args, kwargs, _, args_idxs, kwargs_idxs = create_args_kwargs(
        args_np=args_np,
        arg_np_vals=arg_np_vals,
        args_idxs=args_idxs,
        kwargs_np=kwargs_np,
        kwarg_np_vals=kwarg_np_vals,
        kwargs_idxs=kwargs_idxs,
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable_flags,
        native_array_flags=native_array_flags,
        container_flags=container_flags,
    )
    _, grads_from_gt = ivy.execute_with_gradients(
        grad_fn, [args, kwargs], xs_grad_idxs=xs_grad_idxs, ret_grad_idxs=ret_grad_idxs
    )
    grads_np_from_gt_flat = flatten_and_to_np(ret=grads_from_gt)
    ivy.unset_backend()

    assert len(grads_np_flat) == len(
        grads_np_from_gt_flat
    ), "result length mismatch: {} ({}) != {} ({})".format(
        grads_np_flat,
        len(grads_np_flat),
        grads_np_from_gt_flat,
        len(grads_np_from_gt_flat),
    )

    for grad_np_flat, grad_np_from_gt_flat in zip(grads_np_flat, grads_np_from_gt_flat):
        value_test(
            ret_np_flat=grad_np_flat,
            ret_np_from_gt_flat=grad_np_from_gt_flat,
            rtol=rtol_,
            atol=atol_,
            ground_truth_backend=ground_truth_backend,
        )


def test_method(
    *,
    init_input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]] = None,
    init_as_variable_flags: Union[List[bool], pf.AsVariableFlags] = None,
    init_num_positional_args: Union[int, pf.NumPositionalArg] = 0,
    init_native_array_flags: Union[List[bool], pf.NativeArrayFlags] = None,
    init_all_as_kwargs_np: dict = None,
    method_input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    method_as_variable_flags: Union[List[bool], pf.AsVariableFlags],
    method_num_positional_args: Union[int, pf.NumPositionalArg],
    method_native_array_flags: Union[List[bool], pf.NativeArrayFlags],
    method_container_flags: Union[List[bool], pf.ContainerFlags],
    method_all_as_kwargs_np: dict,
    class_name: str,
    method_name: str = "__call__",
    init_with_v: bool = False,
    method_with_v: bool = False,
    rtol_: float = None,
    atol_: float = 1e-06,
    test_values: Union[bool, str] = True,
    test_gradients: bool = False,
    ground_truth_backend: str,
    device_: str = "cpu",
):
    """Tests a class-method that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    init_input_dtypes
        data types of the input arguments to the constructor in order.
    init_as_variable_flags
        dictates whether the corresponding input argument passed to the constructor
        should be treated as an ivy.Array.
    init_num_positional_args
        number of input arguments that must be passed as positional arguments to the
        constructor.
    init_native_array_flags
        dictates whether the corresponding input argument passed to the constructor
        should be treated as a native array.
    init_all_as_kwargs_np:
        input arguments to the constructor as keyword arguments.
    method_input_dtypes
        data types of the input arguments to the method in order.
    method_as_variable_flags
        dictates whether the corresponding input argument passed to the method should
        be treated as an ivy.Array.
    method_num_positional_args
        number of input arguments that must be passed as positional arguments to the
        method.
    method_native_array_flags
        dictates whether the corresponding input argument passed to the method should
        be treated as a native array.
    method_container_flags
        dictates whether the corresponding input argument passed to the method should
        be treated as an ivy Container.
    method_all_as_kwargs_np:
        input arguments to the method as keyword arguments.
    class_name
        name of the class to test.
    method_name
        name of tthe method to test.
    init_with_v
        if the class being tested is an ivy.Module, then setting this flag as True will
        call the constructor with the variables v passed explicitly.
    method_with_v
        if the class being tested is an ivy.Module, then setting this flag as True will
        call the method with the variables v passed explicitly.
    rtol_
        relative tolerance value.
    atol_
        absolute tolerance value.
    test_values
        can be a bool or a string to indicate whether correctness of values should be
        tested. If the value is `with_v`, shapes are tested but not values.
    ground_truth_backend
        Ground Truth Backend to compare the result-values.
    device_
        The device on which to create arrays.

    Returns
    -------
    ret
        optional, return value from the function
    ret_gt
        optional, return value from the Ground Truth function
    """
    _assert_dtypes_are_valid(method_input_dtypes)
    # split the arguments into their positional and keyword components

    # Constructor arguments #
    (init_input_dtypes, init_as_variable_flags, init_native_array_flags,) = (
        ivy.default(init_input_dtypes, []),
        ivy.default(init_as_variable_flags, []),
        ivy.default(init_native_array_flags, []),
    )
    _assert_dtypes_are_valid(init_input_dtypes)

    init_all_as_kwargs_np = ivy.default(init_all_as_kwargs_np, dict())
    (
        method_input_dtypes,
        method_as_variable_flags,
        method_native_array_flags,
        method_container_flags,
    ) = as_lists(
        method_input_dtypes,
        method_as_variable_flags,
        method_native_array_flags,
        method_container_flags,
    )

    args_np_constructor, kwargs_np_constructor = kwargs_to_args_n_kwargs(
        num_positional_args=init_num_positional_args,
        kwargs=init_all_as_kwargs_np,
    )

    # extract all arrays from the arguments and keyword arguments
    con_arg_np_vals, con_args_idxs, con_c_arg_vals = _get_nested_np_arrays(
        args_np_constructor
    )
    con_kwarg_np_vals, con_kwargs_idxs, con_c_kwarg_vals = _get_nested_np_arrays(
        kwargs_np_constructor
    )

    # make all lists equal in length
    num_arrays_constructor = con_c_arg_vals + con_c_kwarg_vals
    if len(init_input_dtypes) < num_arrays_constructor:
        init_input_dtypes = [
            init_input_dtypes[0] for _ in range(num_arrays_constructor)
        ]
    if len(init_as_variable_flags) < num_arrays_constructor:
        init_as_variable_flags = [
            init_as_variable_flags[0] for _ in range(num_arrays_constructor)
        ]
    if len(init_native_array_flags) < num_arrays_constructor:
        init_native_array_flags = [
            init_native_array_flags[0] for _ in range(num_arrays_constructor)
        ]

    # update variable flags to be compatible with float dtype
    init_as_variable_flags = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(init_as_variable_flags, init_input_dtypes)
    ]

    # Create Args
    args_constructor, kwargs_constructor, _, _, _ = create_args_kwargs(
        args_np=args_np_constructor,
        arg_np_vals=con_arg_np_vals,
        args_idxs=con_args_idxs,
        kwargs_np=kwargs_np_constructor,
        kwarg_np_vals=con_kwarg_np_vals,
        kwargs_idxs=con_kwargs_idxs,
        input_dtypes=init_input_dtypes,
        as_variable_flags=init_as_variable_flags,
        native_array_flags=init_native_array_flags,
    )
    # End constructor #

    # Method arguments #
    args_np_method, kwargs_np_method = kwargs_to_args_n_kwargs(
        num_positional_args=method_num_positional_args, kwargs=method_all_as_kwargs_np
    )

    # extract all arrays from the arguments and keyword arguments
    met_arg_np_vals, met_args_idxs, met_c_arg_vals = _get_nested_np_arrays(
        args_np_method
    )
    met_kwarg_np_vals, met_kwargs_idxs, met_c_kwarg_vals = _get_nested_np_arrays(
        kwargs_np_method
    )

    # make all lists equal in length
    num_arrays_method = met_c_arg_vals + met_c_kwarg_vals
    if len(method_input_dtypes) < num_arrays_method:
        method_input_dtypes = [method_input_dtypes[0] for _ in range(num_arrays_method)]
    if len(method_as_variable_flags) < num_arrays_method:
        method_as_variable_flags = [
            method_as_variable_flags[0] for _ in range(num_arrays_method)
        ]
    if len(method_native_array_flags) < num_arrays_method:
        method_native_array_flags = [
            method_native_array_flags[0] for _ in range(num_arrays_method)
        ]
    if len(method_container_flags) < num_arrays_method:
        method_container_flags = [
            method_container_flags[0] for _ in range(num_arrays_method)
        ]

    method_as_variable_flags = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(method_as_variable_flags, method_input_dtypes)
    ]

    # Create Args
    args_method, kwargs_method, _, _, _ = create_args_kwargs(
        args_np=args_np_method,
        arg_np_vals=met_arg_np_vals,
        args_idxs=met_args_idxs,
        kwargs_np=kwargs_np_method,
        kwarg_np_vals=met_kwarg_np_vals,
        kwargs_idxs=met_kwargs_idxs,
        input_dtypes=method_input_dtypes,
        as_variable_flags=method_as_variable_flags,
        native_array_flags=method_native_array_flags,
        container_flags=method_container_flags,
    )
    # End Method #

    # Run testing
    ins = ivy.__dict__[class_name](*args_constructor, **kwargs_constructor)
    v_np = None
    if isinstance(ins, ivy.Module):
        if init_with_v:
            v = ivy.Container(
                ins._create_variables(device=device_, dtype=method_input_dtypes[0])
            )
            ins = ivy.__dict__[class_name](*args_constructor, **kwargs_constructor, v=v)
        v = ins.__getattribute__("v")
        v_np = v.cont_map(lambda x, kc: ivy.to_numpy(x) if ivy.is_array(x) else x)
        if method_with_v:
            kwargs_method = dict(**kwargs_method, v=v)
    ret, ret_np_flat = get_ret_and_flattened_np_array(
        ins.__getattribute__(method_name), *args_method, **kwargs_method
    )

    # Compute the return with a Ground Truth backend
    ivy.set_backend(ground_truth_backend)
    args_gt_constructor, kwargs_gt_constructor, _, _, _ = create_args_kwargs(
        args_np=args_np_constructor,
        arg_np_vals=con_arg_np_vals,
        args_idxs=con_args_idxs,
        kwargs_np=kwargs_np_constructor,
        kwarg_np_vals=con_kwarg_np_vals,
        kwargs_idxs=con_kwargs_idxs,
        input_dtypes=init_input_dtypes,
        as_variable_flags=init_as_variable_flags,
        native_array_flags=init_native_array_flags,
    )
    args_gt_method, kwargs_gt_method, _, _, _ = create_args_kwargs(
        args_np=args_np_method,
        arg_np_vals=met_arg_np_vals,
        args_idxs=met_args_idxs,
        kwargs_np=kwargs_np_method,
        kwarg_np_vals=met_kwarg_np_vals,
        kwargs_idxs=met_kwargs_idxs,
        input_dtypes=method_input_dtypes,
        as_variable_flags=method_as_variable_flags,
        native_array_flags=method_native_array_flags,
        container_flags=method_container_flags,
    )
    ins_gt = ivy.__dict__[class_name](*args_gt_constructor, **kwargs_gt_constructor)
    if isinstance(ins_gt, ivy.Module):
        v_gt = v_np.cont_map(
            lambda x, kc: ivy.asarray(x) if isinstance(x, np.ndarray) else x
        )
        kwargs_gt_method = dict(**kwargs_gt_method, v=v_gt)
    ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
        ins_gt.__getattribute__(method_name), *args_gt_method, **kwargs_gt_method
    )
    ivy.unset_backend()
    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, ret_from_gt
    # value test

    if isinstance(rtol_, dict):
        rtol_ = _get_framework_rtol(rtol_, ivy.backend)
    if isinstance(atol_, dict):
        atol_ = _get_framework_atol(atol_, ivy.backend)

    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=ret_np_from_gt_flat,
        rtol=rtol_,
        atol=atol_,
    )


def test_frontend_method(
    *,
    init_input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]] = None,
    init_as_variable_flags: Union[List[bool], pf.AsVariableFlags] = None,
    init_num_positional_args: Union[int, pf.NumPositionalArgFn] = 0,
    init_native_array_flags: Union[List[bool], pf.NativeArrayFlags] = None,
    init_all_as_kwargs_np: dict = None,
    method_input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    method_as_variable_flags: Union[List[bool], pf.AsVariableFlags],
    method_num_positional_args: Union[int, pf.NumPositionalArgMethod],
    method_native_array_flags: Union[List[bool], pf.NativeArrayFlags],
    method_all_as_kwargs_np: dict,
    frontend: str,
    frontend_method_data: FrontendMethodData,
    rtol_: float = None,
    atol_: float = 1e-06,
    test_values: Union[bool, str] = True,
):
    """Tests a class-method that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    init_input_dtypes
        data types of the input arguments to the constructor in order.
    init_as_variable_flags
        dictates whether the corresponding input argument passed to the constructor
        should be treated as an ivy.Variable.
    init_num_positional_args
        number of input arguments that must be passed as positional arguments to the
        constructor.
    init_native_array_flags
        dictates whether the corresponding input argument passed to the constructor
        should be treated as a native array.
    init_all_as_kwargs_np:
        input arguments to the constructor as keyword arguments.
    method_input_dtypes
        data types of the input arguments to the method in order.
    method_as_variable_flags
        dictates whether the corresponding input argument passed to the method should
        be treated as a variable.
    method_num_positional_args
        number of input arguments that must be passed as positional arguments to the
        method.
    method_native_array_flags
        dictates whether the corresponding input argument passed to the method should
        be treated as a native array.
    method_all_as_kwargs_np:
        input arguments to the method as keyword arguments.
    frontend
        current frontend (framework).
    rtol_
        relative tolerance value.
    atol_
        absolute tolerance value.
    test_values
        can be a bool or a string to indicate whether correctness of values should be
        tested. If the value is `with_v`, shapes are tested but not values.

    Returns
    -------
    ret
        optional, return value from the function
    ret_gt
        optional, return value from the Ground Truth function
    """
    _assert_dtypes_are_valid(init_input_dtypes)
    _assert_dtypes_are_valid(method_input_dtypes)

    # split the arguments into their positional and keyword components

    # Constructor arguments #
    # convert single values to length 1 lists
    (init_input_dtypes, init_as_variable_flags, init_native_array_flags,) = as_lists(
        ivy.default(init_input_dtypes, []),
        ivy.default(init_as_variable_flags, []),
        ivy.default(init_native_array_flags, []),
    )
    init_all_as_kwargs_np = ivy.default(init_all_as_kwargs_np, dict())
    (
        method_input_dtypes,
        method_as_variable_flags,
        method_native_array_flags,
    ) = as_lists(
        method_input_dtypes,
        method_as_variable_flags,
        method_native_array_flags,
    )

    args_np_constructor, kwargs_np_constructor = kwargs_to_args_n_kwargs(
        num_positional_args=init_num_positional_args,
        kwargs=init_all_as_kwargs_np,
    )

    # extract all arrays from the arguments and keyword arguments
    con_arg_np_vals, con_args_idxs, con_c_arg_vals = _get_nested_np_arrays(
        args_np_constructor
    )
    con_kwarg_np_vals, con_kwargs_idxs, con_c_kwarg_vals = _get_nested_np_arrays(
        kwargs_np_constructor
    )

    # make all lists equal in length
    num_arrays_constructor = con_c_arg_vals + con_c_kwarg_vals
    if len(init_input_dtypes) < num_arrays_constructor:
        init_input_dtypes = [
            init_input_dtypes[0] for _ in range(num_arrays_constructor)
        ]
    if len(init_as_variable_flags) < num_arrays_constructor:
        init_as_variable_flags = [
            init_as_variable_flags[0] for _ in range(num_arrays_constructor)
        ]
    if len(init_native_array_flags) < num_arrays_constructor:
        init_native_array_flags = [
            init_native_array_flags[0] for _ in range(num_arrays_constructor)
        ]

    # update variable flags to be compatible with float dtype
    init_as_variable_flags = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(init_as_variable_flags, init_input_dtypes)
    ]

    # Create Args
    args_constructor, kwargs_constructor, _, _, _ = create_args_kwargs(
        args_np=args_np_constructor,
        arg_np_vals=con_arg_np_vals,
        args_idxs=con_args_idxs,
        kwargs_np=kwargs_np_constructor,
        kwarg_np_vals=con_kwarg_np_vals,
        kwargs_idxs=con_kwargs_idxs,
        input_dtypes=init_input_dtypes,
        as_variable_flags=init_as_variable_flags,
        native_array_flags=init_native_array_flags,
    )
    # End constructor #

    # Method arguments #
    args_np_method, kwargs_np_method = kwargs_to_args_n_kwargs(
        num_positional_args=method_num_positional_args, kwargs=method_all_as_kwargs_np
    )

    # extract all arrays from the arguments and keyword arguments
    met_arg_np_vals, met_args_idxs, met_c_arg_vals = _get_nested_np_arrays(
        args_np_method
    )
    met_kwarg_np_vals, met_kwargs_idxs, met_c_kwarg_vals = _get_nested_np_arrays(
        kwargs_np_method
    )

    # make all lists equal in length
    num_arrays_method = met_c_arg_vals + met_c_kwarg_vals
    if len(method_input_dtypes) < num_arrays_method:
        method_input_dtypes = [method_input_dtypes[0] for _ in range(num_arrays_method)]
    if len(method_as_variable_flags) < num_arrays_method:
        method_as_variable_flags = [
            method_as_variable_flags[0] for _ in range(num_arrays_method)
        ]
    if len(method_native_array_flags) < num_arrays_method:
        method_native_array_flags = [
            method_native_array_flags[0] for _ in range(num_arrays_method)
        ]

    method_as_variable_flags = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(method_as_variable_flags, method_input_dtypes)
    ]

    # Create Args
    args_method, kwargs_method, _, _, _ = create_args_kwargs(
        args_np=args_np_method,
        arg_np_vals=met_arg_np_vals,
        args_idxs=met_args_idxs,
        kwargs_np=kwargs_np_method,
        kwarg_np_vals=met_kwarg_np_vals,
        kwargs_idxs=met_kwargs_idxs,
        input_dtypes=method_input_dtypes,
        as_variable_flags=method_as_variable_flags,
        native_array_flags=method_native_array_flags,
    )
    # End Method #

    args_constructor_ivy, kwargs_constructor_ivy = ivy.args_to_ivy(
        *args_constructor, **kwargs_constructor
    )
    args_method_ivy, kwargs_method_ivy = ivy.args_to_ivy(*args_method, **kwargs_method)
    args_constructor_np = ivy.nested_map(
        args_constructor_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )
    kwargs_constructor_np = ivy.nested_map(
        kwargs_constructor_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )
    args_method_np = ivy.nested_map(
        args_method_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )
    kwargs_method_np = ivy.nested_map(
        kwargs_method_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )

    ivy_frontend_creation_fn = getattr(
        frontend_method_data.ivy_init_module, frontend_method_data.init_name
    )
    # Run testing
    ins = ivy_frontend_creation_fn(*args_constructor, **kwargs_constructor)
    ret, ret_np_flat = get_ret_and_flattened_np_array(
        ins.__getattribute__(frontend_method_data.method_name),
        *args_method,
        **kwargs_method,
    )

    # Compute the return with the native frontend framework
    ivy.set_backend(frontend)
    args_constructor_frontend = ivy.nested_map(
        args_constructor_np,
        lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
    )
    kwargs_constructor_frontend = ivy.nested_map(
        kwargs_constructor_np,
        lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
    )
    args_method_frontend = ivy.nested_map(
        args_method_np,
        lambda x: ivy.native_array(x)
        if isinstance(x, np.ndarray)
        else ivy.as_native_dtype(x)
        if isinstance(x, ivy.Dtype)
        else ivy.as_native_dev(x)
        if isinstance(x, ivy.Device)
        else x,
    )
    kwargs_method_frontend = ivy.nested_map(
        kwargs_method_np,
        lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
    )

    # change ivy dtypes to native dtypes
    if "dtype" in kwargs_method_frontend:
        kwargs_method_frontend["dtype"] = ivy.as_native_dtype(
            kwargs_method_frontend["dtype"]
        )

    # change ivy device to native devices
    if "device" in kwargs_method_frontend:
        kwargs_method_frontend["device"] = ivy.as_native_dev(
            kwargs_method_frontend["device"]
        )
    frontend_creation_fn = getattr(
        frontend_method_data.framework_init_module, frontend_method_data.init_name
    )
    ins_gt = frontend_creation_fn(
        *args_constructor_frontend, **kwargs_constructor_frontend
    )
    frontend_ret = ins_gt.__getattribute__(frontend_method_data.method_name)(
        *args_method_frontend, **kwargs_method_frontend
    )
    if frontend == "tensorflow" and isinstance(frontend_ret, tf.TensorShape):
        frontend_ret_np_flat = [np.asarray(frontend_ret, dtype=np.int32)]
    elif ivy.isscalar(frontend_ret):
        frontend_ret_np_flat = [np.asarray(frontend_ret)]
    else:
        # tuplify the frontend return
        if not isinstance(frontend_ret, tuple):
            frontend_ret = (frontend_ret,)
        frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, ivy.is_native_array)
        frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
        frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
    ivy.unset_backend()

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, frontend_ret

    # value test
    if isinstance(rtol_, dict):
        rtol_ = _get_framework_rtol(rtol_, ivy.backend)
    if isinstance(atol_, dict):
        atol_ = _get_framework_atol(atol_, ivy.backend)

    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=rtol_,
        atol=atol_,
        ground_truth_backend=frontend,
    )


# Helpers
DEFAULT_RTOL = None
DEFAULT_ATOL = 1e-06


def _get_framework_rtol(rtols: dict, current_fw: str):
    if current_fw in rtols.keys():
        return rtols[current_fw]
    return DEFAULT_RTOL


def _get_framework_atol(atols: dict, current_fw: str):
    if current_fw in atols.keys():
        return atols[current_fw]
    return DEFAULT_ATOL


def _get_nested_np_arrays(nest):
    """
    A helper function to search for a NumPy arrays in a nest
    Parameters
    ----------
    nest
        nest to search in.

    Returns
    -------
         Items found, indices, and total number of arrays found
    """
    indices = ivy.nested_argwhere(nest, lambda x: isinstance(x, np.ndarray))
    ret = ivy.multi_index_nest(nest, indices)
    return ret, indices, len(ret)


def create_args_kwargs(
    *,
    args_np,
    arg_np_vals,
    args_idxs,
    kwargs_np,
    kwarg_np_vals,
    kwargs_idxs,
    input_dtypes,
    as_variable_flags,
    native_array_flags=None,
    container_flags=None,
):
    """Creates arguments and keyword-arguments for the function to test.

    Parameters
    ----------
    args_np
        A dictionary of arguments in Numpy.
    kwargs_np
        A dictionary of keyword-arguments in Numpy.
    input_dtypes
        data-types of the input arguments and keyword-arguments.
    as_variable_flags
        A list of booleans. if True for a corresponding input argument, it is called
        as an variable.
    native_array_flags
        if not None, the corresponding argument is called as a Native Array.
    container_flags
        if not None, the corresponding argument is called as an Ivy Container.

    Returns
    -------
    Arguments, Keyword-arguments, number of arguments, and indexes on arguments and
    keyword-arguments.
    """
    # create args
    num_arg_vals = len(arg_np_vals)
    arg_array_vals = [
        ivy.array(x, dtype=d) for x, d in zip(arg_np_vals, input_dtypes[:num_arg_vals])
    ]
    arg_array_vals = [
        _variable(x) if v else x
        for x, v in zip(arg_array_vals, as_variable_flags[:num_arg_vals])
    ]
    if native_array_flags:
        arg_array_vals = [
            ivy.to_native(x) if n else x
            for x, n in zip(arg_array_vals, native_array_flags[:num_arg_vals])
        ]
    if container_flags:
        arg_array_vals = [
            as_cont(x=x) if c else x
            for x, c in zip(arg_array_vals, container_flags[:num_arg_vals])
        ]
    args = ivy.copy_nest(args_np, to_mutable=False)
    ivy.set_nest_at_indices(args, args_idxs, arg_array_vals)

    # create kwargs
    kwarg_array_vals = [
        ivy.array(x, dtype=d)
        for x, d in zip(kwarg_np_vals, input_dtypes[num_arg_vals:])
    ]
    kwarg_array_vals = [
        _variable(x) if v else x
        for x, v in zip(kwarg_array_vals, as_variable_flags[num_arg_vals:])
    ]
    if native_array_flags:
        kwarg_array_vals = [
            ivy.to_native(x) if n else x
            for x, n in zip(kwarg_array_vals, native_array_flags[num_arg_vals:])
        ]
    if container_flags:
        kwarg_array_vals = [
            as_cont(x=x) if c else x
            for x, c in zip(kwarg_array_vals, container_flags[num_arg_vals:])
        ]
    kwargs = ivy.copy_nest(kwargs_np, to_mutable=False)
    ivy.set_nest_at_indices(kwargs, kwargs_idxs, kwarg_array_vals)
    return args, kwargs, num_arg_vals, args_idxs, kwargs_idxs


def convtrue(argument):
    """Convert NativeClass in argument to true framework counter part"""
    if isinstance(argument, NativeClass):
        return argument._native_class
    return argument


def kwargs_to_args_n_kwargs(*, num_positional_args, kwargs):
    """Splits the kwargs into args and kwargs, with the first num_positional_args ported
    to args.
    """
    args = [v for v in list(kwargs.values())[:num_positional_args]]
    kwargs = {k: kwargs[k] for k in list(kwargs.keys())[num_positional_args:]}
    return args, kwargs


def flatten_fw_and_to_np(*, ret, fw):
    """Returns a flattened numpy version of the arrays in ret for a given framework."""
    if not isinstance(ret, tuple):
        ret = (ret,)
    if fw == "jax":
        ret_idxs = ivy.nested_argwhere(
            ret, lambda x: ivy.is_ivy_array(x) or is_jax_native_array(x)
        )
    elif fw == "numpy":
        ret_idxs = ivy.nested_argwhere(
            ret, lambda x: ivy.is_ivy_array(x) or is_numpy_native_array(x)
        )
    elif fw == "tensorflow":
        ret_idxs = ivy.nested_argwhere(
            ret, lambda x: ivy.is_ivy_array(x) or is_tensorflow_native_array(x)
        )
    else:
        ret_idxs = ivy.nested_argwhere(
            ret, lambda x: ivy.is_ivy_array(x) or is_torch_native_array(x)
        )
    if len(ret_idxs) == 0:
        ret_idxs = ivy.nested_argwhere(ret, ivy.isscalar)
        ret_flat = ivy.multi_index_nest(ret, ret_idxs)
        ret_flat = [
            ivy.asarray(x, dtype=ivy.Dtype(str(np.asarray(x).dtype))) for x in ret_flat
        ]
    else:
        ret_flat = ivy.multi_index_nest(ret, ret_idxs)
    # convert the return to NumPy
    ret_np_flat = [ivy.to_numpy(x) for x in ret_flat]
    return ret_np_flat


def flatten(*, ret):
    """Returns a flattened numpy version of the arrays in ret."""
    if not isinstance(ret, tuple):
        ret = (ret,)
    ret_idxs = ivy.nested_argwhere(ret, ivy.is_ivy_array)
    # no ivy array in the returned values, which means it returned scalar
    if len(ret_idxs) == 0:
        ret_idxs = ivy.nested_argwhere(ret, ivy.isscalar)
        ret_flat = ivy.multi_index_nest(ret, ret_idxs)
        ret_flat = [
            ivy.asarray(x, dtype=ivy.Dtype(str(np.asarray(x).dtype))) for x in ret_flat
        ]
    else:
        ret_flat = ivy.multi_index_nest(ret, ret_idxs)
    return ret_flat


def flatten_and_to_np(*, ret):
    # flatten the return
    ret_flat = flatten(ret=ret)
    return [ivy.to_numpy(x) for x in ret_flat]


def get_ret_and_flattened_np_array(fn, *args, **kwargs):
    """
    Runs func with args and kwargs, and returns the result along with its flattened
    version.
    """
    ret = fn(*args, **kwargs)

    def map_fn(x):
        if _is_frontend_array(x):
            return x.ivy_array
        if isinstance(x, ivy.functional.frontends.numpy.ndarray):
            return x.ivy_array
        return x

    ret = ivy.nested_map(ret, map_fn, include_derived={tuple: True})
    return ret, flatten_and_to_np(ret=ret)


def get_frontend_ret(fn, *args, **kwargs):
    ret = fn(*args, **kwargs)
    ret = ivy.nested_map(ret, _frontend_array_to_ivy, include_derived={tuple: True})
    return ret


def args_to_container(array_args):
    array_args_container = ivy.Container({str(k): v for k, v in enumerate(array_args)})
    return array_args_container


def as_lists(*args):
    """Changes the elements in args to be of type list."""
    return (a if isinstance(a, list) else [a] for a in args)


def as_cont(*, x):
    """Returns x as an Ivy Container, containing x at all its leaves."""
    return ivy.Container({"a": x, "b": {"c": x, "d": x}})


def var_fn(x, *, dtype=None, device=None):
    """Returns x as a variable wrapping an Ivy Array with given dtype and device"""
    return _variable(ivy.array(x, dtype=dtype, device=device))


def gradient_incompatible_function(*, fn):
    return (
        not ivy.supports_gradients
        and hasattr(fn, "computes_gradients")
        and fn.computes_gradients
    )


def _is_frontend_array(x):
    return (
        isinstance(x, ndarray)
        or isinstance(x, torch_tensor)
        or isinstance(x, tf_tensor)
        or isinstance(x, DeviceArray)
    )


def _frontend_array_to_ivy(x):
    if (
        isinstance(x, ndarray)
        or isinstance(x, torch_tensor)
        or isinstance(x, tf_tensor)
        or isinstance(x, DeviceArray)
    ):
        return x.ivy_array
    else:
        return x
