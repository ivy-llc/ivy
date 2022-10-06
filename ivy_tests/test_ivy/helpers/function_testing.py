# global
import copy
from typing import Union, List
import numpy as np
import jax
import tensorflow as tf
import importlib
import inspect

# local
import ivy
from ivy_tests.test_ivy.test_frontends import NativeClass
from ivy_tests.test_ivy.test_frontends.test_torch import convtorch
from ivy_tests.test_ivy.test_frontends.test_numpy import convnumpy
from ivy_tests.test_ivy.test_frontends.test_tensorflow import convtensor
from ivy_tests.test_ivy.test_frontends.test_jax import convjax
from ivy.functional.backends.jax.general import is_native_array as is_jax_native_array
from ivy.functional.backends.numpy.general import (
    is_native_array as is_numpy_native_array,
)
from ivy.functional.backends.tensorflow.general import (
    is_native_array as is_tensorflow_native_array,
)
from ivy.functional.backends.torch.general import (
    is_native_array as is_torch_native_array,
)

from .assertions import (
    value_test,
    test_unsupported_function,
    check_unsupported_dtype,
    check_unsupported_device,
    check_unsupported_device_and_dtype,
)


# ToDo, this is temporary until unsupported_dtype is embedded
# into helpers.get_dtypes
def _assert_dtypes_are_valid(input_dtypes: Union[List[ivy.Dtype], List[str]]):
    for dtype in input_dtypes:
        if dtype not in ivy.valid_dtypes:
            raise Exception(f"{dtype} is not a valid data type.")


# Function testing


def test_function(
    *,
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    with_out: bool,
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    container_flags: Union[bool, List[bool]],
    instance_method: bool,
    fw: str,
    fn_name: str,
    rtol_: float = None,
    atol_: float = 1e-06,
    test_values: bool = True,
    test_gradients: bool = False,
    ground_truth_backend: str = "tensorflow",
    device_: str = "cpu",
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
        as an ivy Variable.
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
    device_
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
    test_unsupported = check_unsupported_dtype(
        fn=fn, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
    )
    if not test_unsupported:
        test_unsupported = check_unsupported_device(
            fn=fn, input_device=device_, all_as_kwargs_np=all_as_kwargs_np
        )
    if not test_unsupported:
        test_unsupported = check_unsupported_device_and_dtype(
            fn=fn,
            device=device_,
            input_dtypes=input_dtypes,
            all_as_kwargs_np=all_as_kwargs_np,
        )
    if test_unsupported:
        try:
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
        except Exception:
            return
    else:
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
        if test_unsupported:
            test_unsupported_function(
                fn=instance.__getattribute__(fn_name), args=args, kwargs=kwargs
            )
            return

        ret, ret_np_flat = get_ret_and_flattened_np_array(
            instance.__getattribute__(fn_name), *args, **kwargs
        )
    else:
        if test_unsupported:
            test_unsupported_function(
                fn=ivy.__dict__[fn_name], args=args, kwargs=kwargs
            )
            return
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
        test_unsupported = check_unsupported_dtype(
            fn=fn, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
        )
        # create args
        if test_unsupported:
            try:
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
            except Exception:
                ivy.unset_backend()
                return
        else:
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
        if test_unsupported:
            test_unsupported_function(
                fn=ivy.__dict__[fn_name], args=args, kwargs=kwargs
            )
            ivy.unset_backend()
            return
        ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
            ivy.__dict__[fn_name], *args, **kwargs
        )
    except Exception as e:
        ivy.unset_backend()
        raise e
    ivy.unset_backend()
    # gradient test
    if (
        test_gradients
        and not fw == "numpy"
        and all(as_variable_flags)
        and not any(container_flags)
        and not instance_method
    ):
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
            ground_truth_backend=ground_truth_backend,
        )

    # assuming value test will be handled manually in the test function
    if not test_values:
        if return_flat_np_arrays:
            return ret_np_flat, ret_np_from_gt_flat
        return ret, ret_from_gt
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
    as_variable_flags: Union[bool, List[bool]],
    with_out: bool,
    with_inplace: bool = False,
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    device="cpu",
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
    _assert_dtypes_are_valid(input_dtypes)
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

    # parse function name and frontend submodules (jax.lax, jax.numpy etc.)
    *frontend_submods, fn_tree = fn_tree.split(".")

    # check for unsupported dtypes in backend framework
    function = getattr(ivy.functional.frontends.__dict__[frontend], fn_tree)
    test_unsupported = check_unsupported_dtype(
        fn=function, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
    )

    if not test_unsupported:
        test_unsupported = check_unsupported_device_and_dtype(
            fn=function,
            device=device,
            input_dtypes=input_dtypes,
            all_as_kwargs_np=all_as_kwargs_np,
        )

    # create args
    if test_unsupported:
        try:
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
            args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)
        except Exception:
            return
    else:
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

    # frontend function
    frontend_fn = ivy.functional.frontends.__dict__[frontend].__dict__[fn_tree]

    # check and replace NativeClass object in arguments with ivy counterparts
    convs = {
        "jax": convjax,
        "numpy": convnumpy,
        "tensorflow": convtensor,
        "torch": convtorch,
    }
    if frontend in convs:
        conv = convs[frontend]
        args = ivy.nested_map(args, fn=conv, include_derived=True)
        kwargs = ivy.nested_map(kwargs, fn=conv, include_derived=True)

    # run from the Ivy API directly
    if test_unsupported:
        test_unsupported_function(fn=frontend_fn, args=args, kwargs=kwargs)
        return
    # Make copy for arguments for functions that might use
    # inplace update by default
    copy_kwargs = copy.deepcopy(kwargs)
    copy_args = copy.deepcopy(args)
    ret = frontend_fn(*args, **kwargs)
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
        ret = frontend_fn(*args, **kwargs)
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
        if "inplace" in inspect.getfullargspec(frontend_fn).args:
            # the function provides optional inplace update
            # set inplace update to be True and check
            # if returned reference is inputted reference
            # and if inputted reference's content is correctly updated
            copy_kwargs["inplace"] = True
            first_array = ivy.func_wrapper._get_first_array(*copy_args, **copy_kwargs)
            ret_ = frontend_fn(*copy_args, **copy_kwargs)
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
    backend_returned_scalar = False
    try:
        # check for unsupported dtypes in frontend framework
        function = getattr(ivy.functional.frontends.__dict__[frontend], fn_tree)
        test_unsupported = check_unsupported_dtype(
            fn=function, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
        )

        # create frontend framework args
        args_frontend = ivy.nested_map(
            args_np,
            lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
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

        # check and replace the NativeClass objects in arguments with true counterparts
        args_frontend = ivy.nested_map(
            args_frontend, fn=convtrue, include_derived=True, max_depth=10
        )
        kwargs_frontend = ivy.nested_map(
            kwargs_frontend, fn=convtrue, include_derived=True, max_depth=10
        )

        # compute the return via the frontend framework
        frontend_fw = importlib.import_module(".".join([frontend] + frontend_submods))
        if test_unsupported:
            test_unsupported_function(
                fn=frontend_fw.__dict__[fn_tree],
                args=args_frontend,
                kwargs=kwargs_frontend,
            )
            return
        frontend_ret = frontend_fw.__dict__[fn_tree](*args_frontend, **kwargs_frontend)

        if frontend == "numpy" and not isinstance(frontend_ret, np.ndarray):
            backend_returned_scalar = True
            frontend_ret_np_flat = [np.asarray(frontend_ret)]
        else:
            # tuplify the frontend return
            if not isinstance(frontend_ret, tuple):
                frontend_ret = (frontend_ret,)
            frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, ivy.is_native_array)
            frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
            frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
    except Exception as e:
        ivy.unset_backend()
        raise e
    # unset frontend framework from backend
    ivy.unset_backend()

    if backend_returned_scalar:
        ret_np_flat = ivy.to_numpy([ret])
    else:
        ret_np_flat = flatten_and_to_np(ret=ret)

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, frontend_ret

    # value tests, iterating through each array in the flattened returns
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=rtol,
        atol=atol,
        ground_truth_backend=frontend,
    )


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
    ground_truth_backend: str = "tensorflow",
):
    def grad_fn(xs):
        array_vals = [v for k, v in xs.to_iterator()]
        arg_array_vals = array_vals[0 : len(args_idxs)]
        kwarg_array_vals = array_vals[len(args_idxs) :]
        args_writeable = ivy.copy_nest(args)
        kwargs_writeable = ivy.copy_nest(kwargs)
        ivy.set_nest_at_indices(args_writeable, args_idxs, arg_array_vals)
        ivy.set_nest_at_indices(kwargs_writeable, kwargs_idxs, kwarg_array_vals)
        return ivy.mean(ivy.__dict__[fn_name](*args_writeable, **kwargs_writeable))

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
    arg_array_vals = list(ivy.multi_index_nest(args, args_idxs))
    kwarg_array_vals = list(ivy.multi_index_nest(kwargs, kwargs_idxs))
    xs = args_to_container(arg_array_vals + kwarg_array_vals)
    _, ret_np_flat = get_ret_and_flattened_np_array(
        ivy.execute_with_gradients, grad_fn, xs
    )
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
    arg_array_vals = list(ivy.multi_index_nest(args, args_idxs))
    kwarg_array_vals = list(ivy.multi_index_nest(kwargs, kwargs_idxs))
    xs = args_to_container(arg_array_vals + kwarg_array_vals)
    _, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
        ivy.execute_with_gradients, grad_fn, xs
    )
    ivy.unset_backend()

    assert len(ret_np_flat) == len(
        ret_np_from_gt_flat
    ), "result length mismatch: {} ({}) != {} ({})".format(
        ret_np_flat, len(ret_np_flat), ret_np_from_gt_flat, len(ret_np_from_gt_flat)
    )

    if len(ret_np_flat) < 2:
        return

    grads_np_flat = ret_np_flat[1]
    grads_np_from_gt_flat = ret_np_from_gt_flat[1]
    condition_np_flat = np.isfinite(grads_np_flat)
    grads_np_flat = np.where(
        condition_np_flat, grads_np_flat, np.asarray(0.0, dtype=grads_np_flat.dtype)
    )
    condition_np_from_gt_flat = np.isfinite(grads_np_from_gt_flat)
    grads_np_from_gt_flat = np.where(
        condition_np_from_gt_flat,
        grads_np_from_gt_flat,
        np.asarray(0.0, dtype=grads_np_from_gt_flat.dtype),
    )

    value_test(
        ret_np_flat=grads_np_flat,
        ret_np_from_gt_flat=grads_np_from_gt_flat,
        rtol=rtol_,
        atol=atol_,
    )


def test_method(
    *,
    input_dtypes_init: Union[ivy.Dtype, List[ivy.Dtype]] = None,
    as_variable_flags_init: Union[bool, List[bool]] = None,
    num_positional_args_init: int = 0,
    native_array_flags_init: Union[bool, List[bool]] = None,
    all_as_kwargs_np_init: dict = None,
    input_dtypes_method: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags_method: Union[bool, List[bool]],
    num_positional_args_method: int,
    native_array_flags_method: Union[bool, List[bool]],
    container_flags_method: Union[bool, List[bool]],
    all_as_kwargs_np_method: dict,
    class_name: str,
    method_name: str = "__call__",
    init_with_v: bool = False,
    method_with_v: bool = False,
    rtol_: float = None,
    atol_: float = 1e-06,
    test_values: Union[bool, str] = True,
    test_gradients: bool = False,
    ground_truth_backend: str = "tensorflow",
    device_: str = "cpu",
):
    """Tests a class-method that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    input_dtypes_init
        data types of the input arguments to the constructor in order.
    as_variable_flags_init
        dictates whether the corresponding input argument passed to the constructor
        should be treated as an ivy.Array.
    num_positional_args_init
        number of input arguments that must be passed as positional arguments to the
        constructor.
    native_array_flags_init
        dictates whether the corresponding input argument passed to the constructor
        should be treated as a native array.
    all_as_kwargs_np_init:
        input arguments to the constructor as keyword arguments.
    input_dtypes_method
        data types of the input arguments to the method in order.
    as_variable_flags_method
        dictates whether the corresponding input argument passed to the method should
        be treated as an ivy.Array.
    num_positional_args_method
        number of input arguments that must be passed as positional arguments to the
        method.
    native_array_flags_method
        dictates whether the corresponding input argument passed to the method should
        be treated as a native array.
    container_flags_method
        dictates whether the corresponding input argument passed to the method should
        be treated as an ivy Container.
    all_as_kwargs_np_method:
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
    _assert_dtypes_are_valid(input_dtypes_init)
    _assert_dtypes_are_valid(input_dtypes_method)
    # split the arguments into their positional and keyword components

    # Constructor arguments #
    (input_dtypes_init, as_variable_flags_init, native_array_flags_init,) = (
        ivy.default(input_dtypes_init, []),
        ivy.default(as_variable_flags_init, []),
        ivy.default(native_array_flags_init, []),
    )

    all_as_kwargs_np_init = ivy.default(all_as_kwargs_np_init, dict())
    (
        input_dtypes_method,
        as_variable_flags_method,
        native_array_flags_method,
        container_flags_method,
    ) = as_lists(
        input_dtypes_method,
        as_variable_flags_method,
        native_array_flags_method,
        container_flags_method,
    )

    args_np_constructor, kwargs_np_constructor = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args_init,
        kwargs=all_as_kwargs_np_init,
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
    if len(input_dtypes_init) < num_arrays_constructor:
        input_dtypes_init = [
            input_dtypes_init[0] for _ in range(num_arrays_constructor)
        ]
    if len(as_variable_flags_init) < num_arrays_constructor:
        as_variable_flags_init = [
            as_variable_flags_init[0] for _ in range(num_arrays_constructor)
        ]
    if len(native_array_flags_init) < num_arrays_constructor:
        native_array_flags_init = [
            native_array_flags_init[0] for _ in range(num_arrays_constructor)
        ]

    # update variable flags to be compatible with float dtype
    as_variable_flags_init = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(as_variable_flags_init, input_dtypes_init)
    ]

    # Create Args
    args_constructor, kwargs_constructor, _, _, _ = create_args_kwargs(
        args_np=args_np_constructor,
        arg_np_vals=con_arg_np_vals,
        args_idxs=con_args_idxs,
        kwargs_np=kwargs_np_constructor,
        kwarg_np_vals=con_kwarg_np_vals,
        kwargs_idxs=con_kwargs_idxs,
        input_dtypes=input_dtypes_init,
        as_variable_flags=as_variable_flags_init,
        native_array_flags=native_array_flags_init,
    )
    # End constructor #

    # Method arguments #
    args_np_method, kwargs_np_method = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args_method, kwargs=all_as_kwargs_np_method
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
    if len(input_dtypes_method) < num_arrays_method:
        input_dtypes_method = [input_dtypes_method[0] for _ in range(num_arrays_method)]
    if len(as_variable_flags_method) < num_arrays_method:
        as_variable_flags_method = [
            as_variable_flags_method[0] for _ in range(num_arrays_method)
        ]
    if len(native_array_flags_method) < num_arrays_method:
        native_array_flags_method = [
            native_array_flags_method[0] for _ in range(num_arrays_method)
        ]
    if len(container_flags_method) < num_arrays_method:
        container_flags_method = [
            container_flags_method[0] for _ in range(num_arrays_method)
        ]

    as_variable_flags_method = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(as_variable_flags_method, input_dtypes_method)
    ]

    # Create Args
    args_method, kwargs_method, _, _, _ = create_args_kwargs(
        args_np=args_np_method,
        arg_np_vals=met_arg_np_vals,
        args_idxs=met_args_idxs,
        kwargs_np=kwargs_np_method,
        kwarg_np_vals=met_kwarg_np_vals,
        kwargs_idxs=met_kwargs_idxs,
        input_dtypes=input_dtypes_method,
        as_variable_flags=as_variable_flags_method,
        native_array_flags=native_array_flags_method,
        container_flags=container_flags_method,
    )
    # End Method #

    # Run testing
    ins = ivy.__dict__[class_name](*args_constructor, **kwargs_constructor)
    v_np = None
    if isinstance(ins, ivy.Module):
        if init_with_v:
            v = ivy.Container(
                ins._create_variables(device=device_, dtype=input_dtypes_method[0])
            )
            ins = ivy.__dict__[class_name](*args_constructor, **kwargs_constructor, v=v)
        v = ins.__getattribute__("v")
        v_np = v.map(lambda x, kc: ivy.to_numpy(x) if ivy.is_array(x) else x)
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
        input_dtypes=input_dtypes_init,
        as_variable_flags=as_variable_flags_init,
        native_array_flags=native_array_flags_init,
    )
    args_gt_method, kwargs_gt_method, _, _, _ = create_args_kwargs(
        args_np=args_np_method,
        arg_np_vals=met_arg_np_vals,
        args_idxs=met_args_idxs,
        kwargs_np=kwargs_np_method,
        kwarg_np_vals=met_kwarg_np_vals,
        kwargs_idxs=met_kwargs_idxs,
        input_dtypes=input_dtypes_method,
        as_variable_flags=as_variable_flags_method,
        native_array_flags=native_array_flags_method,
        container_flags=container_flags_method,
    )
    ins_gt = ivy.__dict__[class_name](*args_gt_constructor, **kwargs_gt_constructor)
    if isinstance(ins_gt, ivy.Module):
        v_gt = v_np.map(
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
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=ret_np_from_gt_flat,
        rtol=rtol_,
        atol=atol_,
    )


def test_frontend_method(
    *,
    input_dtypes_init: Union[ivy.Dtype, List[ivy.Dtype]] = None,
    as_variable_flags_init: Union[bool, List[bool]] = None,
    num_positional_args_init: int = 0,
    native_array_flags_init: Union[bool, List[bool]] = None,
    all_as_kwargs_np_init: dict = None,
    input_dtypes_method: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags_method: Union[bool, List[bool]],
    num_positional_args_method: int,
    native_array_flags_method: Union[bool, List[bool]],
    all_as_kwargs_np_method: dict,
    frontend: str,
    class_name: str,
    method_name: str = "__init__",
    rtol_: float = None,
    atol_: float = 1e-06,
    test_values: Union[bool, str] = True,
):
    """Tests a class-method that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    input_dtypes_init
        data types of the input arguments to the constructor in order.
    as_variable_flags_init
        dictates whether the corresponding input argument passed to the constructor
        should be treated as an ivy.Variable.
    num_positional_args_init
        number of input arguments that must be passed as positional arguments to the
        constructor.
    native_array_flags_init
        dictates whether the corresponding input argument passed to the constructor
        should be treated as a native array.
    all_as_kwargs_np_init:
        input arguments to the constructor as keyword arguments.
    input_dtypes_method
        data types of the input arguments to the method in order.
    as_variable_flags_method
        dictates whether the corresponding input argument passed to the method should
        be treated as an ivy.Variable.
    num_positional_args_method
        number of input arguments that must be passed as positional arguments to the
        method.
    native_array_flags_method
        dictates whether the corresponding input argument passed to the method should
        be treated as a native array.
    all_as_kwargs_np_method:
        input arguments to the method as keyword arguments.
    frontend
        current frontend (framework).
    class_name
        name of the class to test.
    method_name
        name of the method to test.
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
    _assert_dtypes_are_valid(input_dtypes_init)
    _assert_dtypes_are_valid(input_dtypes_method)
    ARR_INS_METHOD = {
        "DeviceArray": jax.numpy.array,
        "ndarray": np.array,
        "Tensor": tf.constant,
    }
    # split the arguments into their positional and keyword components

    # Constructor arguments #
    # convert single values to length 1 lists
    (input_dtypes_init, as_variable_flags_init, native_array_flags_init,) = as_lists(
        ivy.default(input_dtypes_init, []),
        ivy.default(as_variable_flags_init, []),
        ivy.default(native_array_flags_init, []),
    )
    all_as_kwargs_np_init = ivy.default(all_as_kwargs_np_init, dict())
    (
        input_dtypes_method,
        as_variable_flags_method,
        native_array_flags_method,
    ) = as_lists(
        input_dtypes_method,
        as_variable_flags_method,
        native_array_flags_method,
    )

    args_np_constructor, kwargs_np_constructor = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args_init,
        kwargs=all_as_kwargs_np_init,
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
    if len(input_dtypes_init) < num_arrays_constructor:
        input_dtypes_init = [
            input_dtypes_init[0] for _ in range(num_arrays_constructor)
        ]
    if len(as_variable_flags_init) < num_arrays_constructor:
        as_variable_flags_init = [
            as_variable_flags_init[0] for _ in range(num_arrays_constructor)
        ]
    if len(native_array_flags_init) < num_arrays_constructor:
        native_array_flags_init = [
            native_array_flags_init[0] for _ in range(num_arrays_constructor)
        ]

    # update variable flags to be compatible with float dtype
    as_variable_flags_init = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(as_variable_flags_init, input_dtypes_init)
    ]

    # Create Args
    args_constructor, kwargs_constructor, _, _, _ = create_args_kwargs(
        args_np=args_np_constructor,
        arg_np_vals=con_arg_np_vals,
        args_idxs=con_args_idxs,
        kwargs_np=kwargs_np_constructor,
        kwarg_np_vals=con_kwarg_np_vals,
        kwargs_idxs=con_kwargs_idxs,
        input_dtypes=input_dtypes_init,
        as_variable_flags=as_variable_flags_init,
        native_array_flags=native_array_flags_init,
    )
    # End constructor #

    # Method arguments #
    args_np_method, kwargs_np_method = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args_method, kwargs=all_as_kwargs_np_method
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
    if len(input_dtypes_method) < num_arrays_method:
        input_dtypes_method = [input_dtypes_method[0] for _ in range(num_arrays_method)]
    if len(as_variable_flags_method) < num_arrays_method:
        as_variable_flags_method = [
            as_variable_flags_method[0] for _ in range(num_arrays_method)
        ]
    if len(native_array_flags_method) < num_arrays_method:
        native_array_flags_method = [
            native_array_flags_method[0] for _ in range(num_arrays_method)
        ]

    as_variable_flags_method = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(as_variable_flags_method, input_dtypes_method)
    ]

    # Create Args
    args_method, kwargs_method, _, _, _ = create_args_kwargs(
        args_np=args_np_method,
        arg_np_vals=met_arg_np_vals,
        args_idxs=met_args_idxs,
        kwargs_np=kwargs_np_method,
        kwarg_np_vals=met_kwarg_np_vals,
        kwargs_idxs=met_kwargs_idxs,
        input_dtypes=input_dtypes_method,
        as_variable_flags=as_variable_flags_method,
        native_array_flags=native_array_flags_method,
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

    # Run testing
    class_name = class_name.split(".")
    ins_class = ivy.functional.frontends.__dict__[frontend]
    if class_name[-1] in ARR_INS_METHOD and frontend != "torch":
        frontend_class = ARR_INS_METHOD[class_name[-1]]
        for c_n in class_name:
            ins_class = getattr(ins_class, c_n)
    else:
        frontend_class = importlib.import_module(frontend)
        for c_n in class_name:
            ins_class = getattr(ins_class, c_n)
            frontend_class = getattr(frontend_class, c_n)
    ins = ins_class(*args_constructor, **kwargs_constructor)
    ret, ret_np_flat = get_ret_and_flattened_np_array(
        ins.__getattribute__(method_name), *args_method, **kwargs_method
    )

    # Compute the return with the native frontend framework
    ivy.set_backend(frontend)
    backend_returned_scalar = False
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
        lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
    )
    kwargs_method_frontend = ivy.nested_map(
        kwargs_method_np,
        lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
    )

    ins_gt = frontend_class(*args_constructor_frontend, **kwargs_constructor_frontend)
    frontend_ret = ins_gt.__getattribute__(method_name)(
        *args_method_frontend, **kwargs_method_frontend
    )
    if frontend == "numpy" and not isinstance(frontend_ret, np.ndarray):
        backend_returned_scalar = True
        frontend_ret_np_flat = [np.asarray(frontend_ret)]
    elif frontend == "tensorflow" and not isinstance(frontend_ret, tf.Tensor):
        frontend_ret_np_flat = [ivy.array(frontend_ret).to_numpy()]
    else:
        # tuplify the frontend return
        if not isinstance(frontend_ret, tuple):
            frontend_ret = (frontend_ret,)
        frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, ivy.is_native_array)
        frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
        frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
    ivy.unset_backend()

    if backend_returned_scalar:
        ret_np_flat = ivy.to_numpy([ret])
    else:
        ret_np_flat = flatten_and_to_np(ret=ret)

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, frontend_ret
    # value test
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=rtol_,
        atol=atol_,
        ground_truth_backend=frontend,
    )


def test_frontend_array_instance_method(
    *,
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    with_out: bool,
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    frontend: str,
    frontend_class: object,
    fn_tree: str,
    rtol: float = None,
    atol: float = 1e-06,
    test_values: bool = True,
    **all_as_kwargs_np,
):
    """Tests a frontend instance method for the current backend by comparing the
    result with the function in the associated framework.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated
        as an ivy Variable.
    with_out
        if True, the function is also tested with the optional out argument.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    native_array_flags
        dictates whether the corresponding input argument should be treated
        as a native array.
    frontend
        current frontend (framework).
    frontend_class
        class in the frontend framework.
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
    # num_positional_args ignores self, which we need to compensate for
    num_positional_args += 1

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

    # update variable flags to be compatible with float dtype and with_out args
    as_variable_flags = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]

    # parse function name and frontend submodules (i.e. jax.lax, jax.numpy etc.)
    *frontend_submods, fn_tree = fn_tree.split(".")

    # check for unsupported dtypes in backend framework
    function = getattr(ivy.functional.frontends.__dict__[frontend], fn_tree)
    test_unsupported = check_unsupported_dtype(
        fn=function, input_dtypes=input_dtypes, all_as_kwargs_np=all_as_kwargs_np
    )

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(
        num_positional_args=num_positional_args, kwargs=all_as_kwargs_np
    )

    # create args
    if test_unsupported:
        try:
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
            args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)
        except Exception:
            return
    else:
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
        args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)

    # get instance array
    if args == []:
        instance_array = list(kwargs.values())[0]
        del kwargs[(list(kwargs.keys())[0])]
    else:
        instance_array = args[0]
        args = args[1:]

    # create class instance
    class_instance = frontend_class(instance_array)

    # frontend function
    fn_name = fn_tree.split(".")[-1]
    frontend_fn = class_instance.__getattribute__(fn_name)

    # run from Ivy API directly
    if test_unsupported:
        test_unsupported_function(fn=frontend_fn, args=args, kwargs=kwargs)
        return

    ret = frontend_fn(*args, **kwargs)
    ret = ivy.array(ret) if with_out and not ivy.is_array(ret) else ret

    # assert idx of return if the idx of the out array provided
    out = ret
    if with_out:
        assert not isinstance(ret, tuple)
        assert ivy.is_array(ret)
        if "out" in kwargs:
            kwargs["out"] = out
            kwargs_ivy["out"] = ivy.asarray(out)  # case where ret is not ivy.array
        else:
            args[ivy.arg_info(frontend_fn, name="out")["idx"]] = out
            args_ivy = list(args_ivy)
            args_ivy[ivy.arg_info(frontend_fn, name="out")["idx"]] = ivy.asarray(
                out
            )  # case where ret is not ivy.array
            args_ivy = tuple(args_ivy)
        ret = frontend_fn(*args, **kwargs)

        if ivy.native_inplace_support:
            # these backends do not always support native inplace updates
            assert ret.data is out.data

    # create NumPy args
    args_np = ivy.nested_map(
        args_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )
    kwargs_np = ivy.nested_map(
        kwargs_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )

    # get instance array
    if args_np == [] or args_np == ():
        instance_np_array = list(kwargs_np.values())[0]
    else:
        instance_np_array = args_np[0]

    # create class instance
    class_instance_np = frontend_class(instance_np_array)

    # frontend function
    frontend_fn_np = class_instance_np.__getattribute__(fn_name)

    # remove self from all_as_kwargs_np
    del all_as_kwargs_np[(list(kwargs_np.keys())[0])]

    # temporarily set frontend framework as backend
    ivy.set_backend(frontend)
    backend_returned_scalar = False
    try:
        # run from Ivy API directly
        test_unsupported = check_unsupported_dtype(
            fn=frontend_fn_np,
            input_dtypes=input_dtypes,
            all_as_kwargs_np=all_as_kwargs_np,
        )

        # create frontend framework args
        args_frontend = ivy.nested_map(
            args_np,
            lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
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

        # change out argument to ivy array
        if "out" in kwargs_frontend:
            if kwargs_frontend["out"] is not None:
                kwargs_frontend["out"] = ivy.asarray(kwargs_frontend["out"])

        # get instance array
        if args_frontend == () or args_frontend == []:
            frontend_instance_array = list(kwargs_frontend.values())[0]
            del kwargs_frontend[(list(kwargs_frontend.keys())[0])]
        else:
            frontend_instance_array = args_frontend[0]
            args_frontend = args_frontend[1:]

        # create class instance
        frontend_class_instance = frontend_class(frontend_instance_array)

        # frontend function
        frontend_fn = frontend_class_instance.__getattribute__(fn_name)

        # return from frontend framework
        if test_unsupported:
            test_unsupported_function(
                fn=frontend_fn, args=args_frontend, kwargs=kwargs_frontend
            )
            return
        frontend_ret = frontend_fn(*args_frontend, **kwargs_frontend)

        if frontend == "numpy" and not isinstance(frontend_ret, np.ndarray):
            backend_returned_scalar = True
            frontend_ret_np_flat = [np.asarray(frontend_ret)]
        else:
            # tuplify the frontend return
            if not isinstance(frontend_ret, tuple):
                frontend_ret = (frontend_ret,)
            frontend_ret_idxs = ivy.nested_argwhere(frontend_ret, ivy.is_array)
            frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
            frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]
    except Exception as e:
        ivy.unset_backend()
        raise e
    # unset frontend framework from backend
    ivy.unset_backend()

    # handle scalar return
    if backend_returned_scalar:
        ret_np_flat = ivy.to_numpy([ret])
    else:
        ret_np_flat = flatten_and_to_np(ret=ret)

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, frontend_ret

    # value tests, iterating through each array in the flattened returns
    value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=rtol,
        atol=atol,
        ground_truth_backend=frontend,
    )


# Helpers


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
        as an Ivy Variable.
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
        ivy.variable(x) if v else x
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
        ivy.variable(x) if v else x
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


def flatten_fw(*, ret, fw):
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
        ret_flat = [ivy.asarray(x) for x in ret_flat]
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
    return ret, flatten_and_to_np(ret=ret)


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
    """Returns x as an Ivy Variable wrapping an Ivy Array with given dtype and device"""
    return ivy.variable(ivy.array(x, dtype=dtype, device=device))


def gradient_incompatible_function(*, fn):
    return (
        not ivy.supports_gradients
        and hasattr(fn, "computes_gradients")
        and fn.computes_gradients
    )
