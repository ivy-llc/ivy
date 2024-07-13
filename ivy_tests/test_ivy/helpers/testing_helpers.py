# general
import json
import os
import pytest
import importlib
import inspect
import functools
from typing import List, Optional

from hypothesis import given, strategies as st, example

# local
import ivy.functional.frontends.numpy as np_frontend
from .hypothesis_helpers import number_helpers as nh
from .globals import TestData
from . import test_parameter_flags as pf
from . import test_globals as t_globals
from .pipeline_helper import BackendHandler
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    DynamicFlag,
    BuiltInstanceStrategy,
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
    BuiltGradientStrategy,
    BuiltContainerStrategy,
    BuiltWithOutStrategy,
    BuiltWithCopyStrategy,
    BuiltInplaceStrategy,
    BuiltTraceStrategy,
    BuiltTraceEachStrategy,
    BuiltFrontendArrayStrategy,
    BuiltTranspileStrategy,
    BuiltPrecisionModeStrategy,
    BuiltCythonWrapperStrategy,
)
from ivy_tests.test_ivy.helpers.structs import FrontendMethodData
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import (
    _dtype_kind_keys,
    _get_type_dict,
)
from .globals import mod_backend

cmd_line_args = (
    "with_out",
    "instance_method",
    "test_gradients",
    "test_trace",
    "test_trace_each",
    "precision_mode",
)
cmd_line_args_lists = (
    "as_variable",
    "native_array",
    "container",
)


def _get_runtime_flag_value(flag):
    return flag.strategy if isinstance(flag, DynamicFlag) else flag


@st.composite
def num_positional_args_method(draw, *, method):
    """Draws an integers randomly from the minimum and maximum number of
    positional arguments a given method can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    method
        callable method

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.
    """
    total, num_positional_only, num_keyword_only = (0, 0, 0)
    for param in inspect.signature(method).parameters.values():
        if param.name == "self":
            continue
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind in [param.KEYWORD_ONLY, param.VAR_KEYWORD]:
            num_keyword_only += 1
    return draw(
        nh.ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


@st.composite
def num_positional_args(draw, *, fn_name: Optional[str] = None):
    """Draws an integers randomly from the minimum and maximum number of
    positional arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a
        given data-set (ex. list).
    fn_name
        name of the function.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.

    Examples
    --------
    @given(
        num_positional_args=num_positional_args(fn_name="floor_divide")
    )
    @given(
        num_positional_args=num_positional_args(fn_name="add")
    )
    """
    if mod_backend[t_globals.CURRENT_BACKEND]:
        proc, input_queue, output_queue = mod_backend[t_globals.CURRENT_BACKEND]
        input_queue.put(
            ("num_positional_args_helper", fn_name, t_globals.CURRENT_BACKEND)
        )
        num_positional_only, total, num_keyword_only = output_queue.get()
    else:
        num_positional_only, total, num_keyword_only = num_positional_args_helper(
            fn_name, t_globals.CURRENT_BACKEND
        )
    return draw(
        nh.ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


def num_positional_args_helper(fn_name, backend):
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    fn = None
    with BackendHandler.update_backend(backend) as ivy_backend:
        ivy_backend.utils.dynamic_import.import_module(fn_name.rpartition(".")[0])
        for i, fn_name_key in enumerate(fn_name.split(".")):
            if i == 0:
                fn = ivy_backend.__dict__[fn_name_key]
            else:
                fn = fn.__dict__[fn_name_key]
    for param in inspect.signature(fn).parameters.values():
        if param.name == "self":
            continue
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind in [param.KEYWORD_ONLY, param.VAR_KEYWORD]:
            num_keyword_only += 1
    return num_positional_only, total, num_keyword_only


# Decorators helpers


def _import_fn(fn_tree: str):
    """Import a function from function tree string.

    Parameters
    ----------
    fn_tree
        Full function tree without "ivy" root
        example: "functional.backends.jax.creation.arange".

    Returns
    -------
    Returns fn_name, imported module, callable function
    """
    split_index = fn_tree.rfind(".")
    fn_name = fn_tree[split_index + 1 :]
    module_to_import = fn_tree[:split_index]
    mod = importlib.import_module(module_to_import)
    try:
        callable_fn = mod.__dict__[fn_name]
    except KeyError:
        raise ImportError(
            f"Error: The function '{fn_name}' could not be found within the module"
            f" '{module_to_import}'.\nPlease double-check the function name and its"
            " associated path.\nIf this function is a new feature you'd like to see,"
            " we'd love to hear from you! You can contribute to our project. For more"
            " details, please"
            " visit:\nhttps://lets-unify.ai/ivy/contributing/open_tasks.html\n"
        )
    return callable_fn, fn_name, module_to_import


def _get_method_supported_devices_dtypes_helper(
    method_name: str, class_module: str, class_name: str, backend_str: str
):
    # helper to delegate backend related
    # computation outside the main function
    # so as to ease multiprocessing
    with BackendHandler.update_backend(backend_str) as backend:
        _fn = getattr(class_module.__dict__[class_name], method_name)
        devices_and_dtypes = backend.function_supported_devices_and_dtypes(_fn)
        organized_dtypes = {}
        for device in devices_and_dtypes.keys():
            organized_dtypes[device] = _partition_dtypes_into_kinds(
                backend_str, devices_and_dtypes[device]
            )
    return organized_dtypes


def _get_method_supported_devices_dtypes(
    method_name: str, class_module: str, class_name: str
):
    """Get supported devices and data types for a method in Ivy API.

    Parameters
    ----------
    method_name
        Name of the method in the class

    class_module
        Name of the class module

    class_name
        Name of the class

    Returns
    -------
    Returns a dictionary containing supported device types and its supported data types
    for the method
    """
    supported_device_dtypes = {}

    for backend_str in available_frameworks:
        if mod_backend[backend_str]:
            # we gotta do this using multiprocessing
            proc, input_queue, output_queue = mod_backend[backend_str]
            input_queue.put(
                (
                    "method supported dtypes",
                    method_name,
                    class_module.__name__,
                    class_name,
                    backend_str,
                )
            )
            supported_device_dtypes[backend_str] = output_queue.get()
        else:
            supported_device_dtypes[backend_str] = (
                _get_method_supported_devices_dtypes_helper(
                    method_name, class_module, class_name, backend_str
                )
            )
    return supported_device_dtypes


def _get_supported_devices_dtypes_helper(
    backend_str: str, fn_module: str, fn_name: str
):
    # helper function so as to ease multiprocessing
    with BackendHandler.update_backend(backend_str) as backend:
        _tmp_mod = importlib.import_module(fn_module)  # TODO use dynamic import?
        _fn = _tmp_mod.__dict__[fn_name]
        devices_and_dtypes = backend.function_supported_devices_and_dtypes(_fn)
        try:
            # Issue with bfloat16 and tensorflow
            if "bfloat16" in devices_and_dtypes["gpu"]:
                tmp = list(devices_and_dtypes["gpu"])
                tmp.remove("bfloat16")
                devices_and_dtypes["gpu"] = tuple(tmp)
        except KeyError:
            pass
        organized_dtypes = {}
        for device in devices_and_dtypes.keys():
            organized_dtypes[device] = _partition_dtypes_into_kinds(
                backend_str, devices_and_dtypes[device]
            )
    return organized_dtypes


def _get_supported_devices_dtypes(fn_name: str, fn_module: str):
    """Get supported devices and data types for a function in Ivy API.

    Parameters
    ----------
    fn_name
        Name of the function

    fn_module
        Full import path of the function module

    Returns
    -------
    Returns a dictionary containing supported device types and its supported data types
    for the function
    """
    supported_device_dtypes = {}

    # This is for getting a private function from numpy frontend where we have
    # a ufunc object as we can't refer to them as functions
    if fn_module == "ivy.functional.frontends.numpy":
        fn_module_ = np_frontend
        if isinstance(getattr(fn_module_, fn_name), fn_module_.ufunc):
            fn_name = f"_{fn_name}"

    for backend_str in available_frameworks:
        if mod_backend[backend_str]:
            # we know we need to use multiprocessing
            # to get the devices and dtypes
            proc, input_queue, output_queue = mod_backend[backend_str]
            input_queue.put(("supported dtypes", fn_module, fn_name, backend_str))
            supported_device_dtypes[backend_str] = output_queue.get()
        else:
            supported_device_dtypes[backend_str] = _get_supported_devices_dtypes_helper(
                backend_str, fn_module, fn_name
            )
    return supported_device_dtypes


def _partition_dtypes_into_kinds(framework: str, dtypes):
    partitioned_dtypes = {}
    for kind in _dtype_kind_keys:
        partitioned_dtypes[kind] = set(_get_type_dict(framework, kind)).intersection(
            dtypes
        )
    return partitioned_dtypes


# Decorators


def handle_test(
    *,
    fn_tree: Optional[str] = None,
    ground_truth_backend: str = "tensorflow",
    number_positional_args=None,
    test_instance_method=BuiltInstanceStrategy,
    test_with_out=BuiltWithOutStrategy,
    test_with_copy=BuiltWithCopyStrategy,
    test_gradients=BuiltGradientStrategy,
    test_trace=BuiltTraceStrategy,
    test_trace_each=BuiltTraceEachStrategy,
    transpile=BuiltTranspileStrategy,
    precision_mode=BuiltPrecisionModeStrategy,
    as_variable_flags=BuiltAsVariableStrategy,
    native_array_flags=BuiltNativeArrayStrategy,
    container_flags=BuiltContainerStrategy,
    test_cython_wrapper=BuiltCythonWrapperStrategy,
    **_given_kwargs,
):
    """Test wrapper for Ivy functions.

    The wrapper sets the required test globals and creates test flags strategies.

    Parameters
    ----------
    fn_tree
        Full function import path

    ground_truth_backend
        The framework to assert test results are equal to

    number_positional_args
        A search strategy for determining the number of positional arguments to be
        passed to the function

    test_instance_method
        A search strategy that generates a boolean to test instance methods

    test_with_out
        A search strategy that generates a boolean to test the function with an `out`
        parameter

    test_with_copy
        A search strategy that generates a boolean to test the function with an `copy`
        parameter

    test_gradients
        A search strategy that generates a boolean to test the function with arrays as
        gradients

    test_trace
        A search strategy that generates a boolean to trace and test the
        function

    test_trace_each
        A search strategy that generates a boolean to trace and test the
        function (trace each example separately)

    precision_mode
        A search strategy that generates a boolean to switch between two different
        precision modes supported by numpy and (torch, jax) and test the function

    as_variable_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a Variable array

    native_array_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a native array

    container_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a Container
    """
    is_fn_tree_provided = fn_tree is not None
    if is_fn_tree_provided:
        fn_tree = f"ivy.{fn_tree}"
    is_hypothesis_test = len(_given_kwargs) != 0

    possible_arguments = {}
    if is_hypothesis_test and is_fn_tree_provided:
        # Use the default strategy
        if number_positional_args is None:
            number_positional_args = num_positional_args(fn_name=fn_tree)
        # Generate the test flags strategy
        possible_arguments["test_flags"] = pf.function_flags(
            ground_truth_backend=st.just(ground_truth_backend),
            num_positional_args=number_positional_args,
            instance_method=_get_runtime_flag_value(test_instance_method),
            with_out=_get_runtime_flag_value(test_with_out),
            with_copy=_get_runtime_flag_value(test_with_copy),
            test_gradients=_get_runtime_flag_value(test_gradients),
            test_trace=_get_runtime_flag_value(test_trace),
            test_trace_each=_get_runtime_flag_value(test_trace_each),
            transpile=_get_runtime_flag_value(transpile),
            as_variable=_get_runtime_flag_value(as_variable_flags),
            native_arrays=_get_runtime_flag_value(native_array_flags),
            container_flags=_get_runtime_flag_value(container_flags),
            precision_mode=_get_runtime_flag_value(precision_mode),
            test_cython_wrapper=_get_runtime_flag_value(test_cython_wrapper),
        )

    def test_wrapper(test_fn):
        if is_fn_tree_provided:
            callable_fn, fn_name, fn_mod = _import_fn(fn_tree)
            supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)
            possible_arguments["fn_name"] = st.just(fn_name)

        # If a test is not a Hypothesis test, we only set the test global data
        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            # Check if these arguments are being asked for
            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                _given_kwargs[key] = possible_arguments[key]
            # Wrap the test with the @given decorator

            hypothesis_test_fn = given(**_given_kwargs)(test_fn)

            @functools.wraps(hypothesis_test_fn)
            def wrapped_test(*args, **kwargs):
                try:
                    hypothesis_test_fn(*args, **kwargs)
                except Exception as e:
                    # A string matching is used instead of actual exception due to
                    # exception object in with_backend is different from global Ivy
                    if e.__class__.__qualname__ == "IvyNotImplementedException":
                        pytest.skip("Function not implemented in backend.")
                    else:
                        raise e

        else:
            wrapped_test = test_fn

        # Set the test data to be used by test helpers
        if is_fn_tree_provided:
            wrapped_test.test_data = TestData(
                test_fn=wrapped_test,
                fn_tree=fn_tree,
                fn_name=fn_name,
                supported_device_dtypes=supported_device_dtypes,
            )
        wrapped_test._ivy_test = True
        wrapped_test.ground_truth_backend = ground_truth_backend

        return wrapped_test

    return test_wrapper


def handle_frontend_test(
    *,
    fn_tree: str,
    gt_fn_tree: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    number_positional_args=None,
    test_with_out=BuiltWithOutStrategy,
    test_with_copy=BuiltWithCopyStrategy,
    test_inplace=BuiltInplaceStrategy,
    as_variable_flags=BuiltAsVariableStrategy,
    native_array_flags=BuiltNativeArrayStrategy,
    test_trace=BuiltTraceStrategy,
    test_trace_each=BuiltTraceEachStrategy,
    generate_frontend_arrays=BuiltFrontendArrayStrategy,
    transpile=BuiltTranspileStrategy,
    precision_mode=BuiltPrecisionModeStrategy,
    **_given_kwargs,
):
    """Test wrapper for Ivy frontend functions.

    The wrapper sets the required test globals and creates test flags strategies.

    Parameters
    ----------
    fn_tree
        Full function import path
    gt_fn_tree
        Full function import path for the ground truth function, by default will be
        the same as fn_tree
    number_positional_args
        A search strategy for determining the number of positional arguments to be
        passed to the function

    test_inplace
        A search strategy that generates a boolean to test the method with `inplace`
        update

    test_with_out
        A search strategy that generates a boolean to test the function with an `out`
        parameter

    test_with_copy
        A search strategy that generates a boolean to test the function with an `copy`
        parameter

    precision_mode
        A search strategy that generates a boolean to switch between two different
        precision modes supported by numpy and (torch, jax) and test the function

    as_variable_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a Variable array

    native_array_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a native array

    test_trace
        A search strategy that generates a boolean to trace and test the
        function

    test_trace_each
        A search strategy that generates a boolean to trace and test the
        function (trace each example separately)

    generate_frontend_arrays
        A search strategy that generates a list of boolean flags for array inputs to
        be frontend array
    """
    fn_tree = f"ivy.functional.frontends.{fn_tree}"
    if aliases is not None:
        for i in range(len(aliases)):
            aliases[i] = f"ivy.functional.frontends.{aliases[i]}"
    is_hypothesis_test = len(_given_kwargs) != 0

    if is_hypothesis_test:
        # Use the default strategy
        if number_positional_args is None:
            number_positional_args = num_positional_args(fn_name=fn_tree)
        # Generate the test flags strategy
        test_flags = pf.frontend_function_flags(
            num_positional_args=number_positional_args,
            with_out=_get_runtime_flag_value(test_with_out),
            with_copy=_get_runtime_flag_value(test_with_copy),
            inplace=_get_runtime_flag_value(test_inplace),
            as_variable=_get_runtime_flag_value(as_variable_flags),
            native_arrays=_get_runtime_flag_value(native_array_flags),
            test_trace=_get_runtime_flag_value(test_trace),
            test_trace_each=_get_runtime_flag_value(test_trace_each),
            generate_frontend_arrays=_get_runtime_flag_value(generate_frontend_arrays),
            transpile=_get_runtime_flag_value(transpile),
            precision_mode=_get_runtime_flag_value(precision_mode),
        )

    def test_wrapper(test_fn):
        callable_fn, fn_name, fn_mod = _import_fn(fn_tree)
        supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)

        # If a test is not a Hypothesis test, we only set the test global data
        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            # Check if these arguments are being asked for
            possible_arguments = {
                "test_flags": test_flags,
                "fn_tree": (
                    st.sampled_from([fn_tree] + aliases)
                    if aliases is not None
                    else st.just(fn_tree)
                ),
                "gt_fn_tree": st.just(gt_fn_tree),
            }
            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                # extend Hypothesis given kwargs with our strategies
                _given_kwargs[key] = possible_arguments[key]
            # Wrap the test with the @given decorator
            hypothesis_test_fn = given(**_given_kwargs)(test_fn)

            @functools.wraps(hypothesis_test_fn)
            def wrapped_test(*args, **kwargs):
                try:
                    hypothesis_test_fn(*args, **kwargs)
                except Exception as e:
                    # A string matching is used instead of actual exception due to
                    # exception object in with_backend is different from global Ivy
                    if e.__class__.__qualname__ == "IvyNotImplementedException":
                        pytest.skip("Function not implemented in backend.")
                    else:
                        raise e

        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=fn_tree,
            fn_name=fn_name,
            supported_device_dtypes=supported_device_dtypes,
        )

        return wrapped_test

    return test_wrapper


def _import_method(method_tree: str):
    split_index = method_tree.rfind(".")
    class_tree, method_name = method_tree[:split_index], method_tree[split_index + 1 :]
    split_index = class_tree.rfind(".")
    mod_to_import, class_name = class_tree[:split_index], class_tree[split_index + 1 :]
    _mod = importlib.import_module(mod_to_import)
    _class = _mod.__getattribute__(class_name)
    _method = getattr(_class, method_name)
    return _method, method_name, _class, class_name, _mod


def handle_method(
    *,
    init_tree: str = "",
    method_tree: Optional[str] = None,
    ground_truth_backend: str = "tensorflow",
    test_gradients=BuiltGradientStrategy,
    test_trace=BuiltTraceStrategy,
    test_trace_each=BuiltTraceEachStrategy,
    precision_mode=BuiltPrecisionModeStrategy,
    init_num_positional_args=None,
    init_native_arrays=BuiltNativeArrayStrategy,
    init_as_variable_flags=BuiltAsVariableStrategy,
    method_num_positional_args=None,
    method_native_arrays=BuiltNativeArrayStrategy,
    method_as_variable_flags=BuiltAsVariableStrategy,
    method_container_flags=BuiltContainerStrategy,
    **_given_kwargs,
):
    """Test wrapper for Ivy methods.

    The wrapper sets the required test globals and creates test flags strategies.

    Parameters
    ----------
    method_tree
        Full method import path

    ground_truth_backend
        The framework to assert test results are equal to
    """
    # need to fill up the docstring
    is_method_tree_provided = method_tree is not None
    if is_method_tree_provided:
        method_tree = f"ivy.{method_tree}"
    is_hypothesis_test = len(_given_kwargs) != 0
    possible_arguments = {
        "ground_truth_backend": st.just(ground_truth_backend),
        "test_gradients": _get_runtime_flag_value(test_gradients),
        "test_trace": _get_runtime_flag_value(test_trace),
        "test_trace_each": _get_runtime_flag_value(test_trace_each),
        "precision_mode": _get_runtime_flag_value(precision_mode),
    }

    if is_hypothesis_test and is_method_tree_provided:
        callable_method, method_name, _, class_name, method_mod = _import_method(
            method_tree
        )

        if init_num_positional_args is None:
            init_num_positional_args = num_positional_args(fn_name=init_tree)

        possible_arguments["init_flags"] = pf.init_method_flags(
            num_positional_args=init_num_positional_args,
            as_variable=_get_runtime_flag_value(init_as_variable_flags),
            native_arrays=_get_runtime_flag_value(init_native_arrays),
            precision_mode=_get_runtime_flag_value(precision_mode),
        )

        if method_num_positional_args is None:
            method_num_positional_args = num_positional_args_method(
                method=callable_method
            )

        possible_arguments["method_flags"] = pf.method_flags(
            num_positional_args=method_num_positional_args,
            as_variable=_get_runtime_flag_value(method_as_variable_flags),
            native_arrays=_get_runtime_flag_value(method_native_arrays),
            container_flags=_get_runtime_flag_value(method_container_flags),
            precision_mode=_get_runtime_flag_value(precision_mode),
        )

    def test_wrapper(test_fn):
        if is_method_tree_provided:
            supported_device_dtypes = _get_method_supported_devices_dtypes(
                method_name, method_mod, class_name
            )
            possible_arguments["class_name"] = st.just(class_name)
            possible_arguments["method_name"] = st.just(method_name)

        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            filtered_args = set(param_names).intersection(possible_arguments.keys())

            for key in filtered_args:
                # extend Hypothesis given kwargs with our strategies
                _given_kwargs[key] = possible_arguments[key]

            hypothesis_test_fn = given(**_given_kwargs)(test_fn)

            @functools.wraps(hypothesis_test_fn)
            def wrapped_test(*args, **kwargs):
                try:
                    hypothesis_test_fn(*args, **kwargs)
                except Exception as e:
                    # A string matching is used instead of actual exception due to
                    # exception object in with_backend is different from global Ivy
                    if e.__class__.__qualname__ == "IvyNotImplementedException":
                        pytest.skip("Function not implemented in backend.")
                    else:
                        raise e

        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=method_tree,
            fn_name=method_name,
            supported_device_dtypes=supported_device_dtypes,
            is_method=True,
        )

        wrapped_test.ground_truth_backend = ground_truth_backend
        wrapped_test._ivy_test = True

        return wrapped_test

    return test_wrapper


def handle_frontend_method(
    *,
    class_tree: str,
    init_tree: str,
    method_name: str,
    init_num_positional_args=None,
    init_native_arrays=BuiltNativeArrayStrategy,
    init_as_variable_flags=BuiltAsVariableStrategy,
    test_trace=BuiltTraceStrategy,
    test_trace_each=BuiltTraceEachStrategy,
    precision_mode=BuiltPrecisionModeStrategy,
    method_num_positional_args=None,
    method_native_arrays=BuiltNativeArrayStrategy,
    method_as_variable_flags=BuiltAsVariableStrategy,
    test_inplace=BuiltInplaceStrategy,
    generate_frontend_arrays=BuiltFrontendArrayStrategy,
    **_given_kwargs,
):
    """Test wrapper for Ivy frontends methods.

    The wrapper sets the required test globals and creates
    test flags strategies.

    Parameters
    ----------
    class_tree
        Full class import path

    init_tree
        Full import path for the function used to create the class

    method_name
        Name of the method

    init_num_positional_args
        A search strategy that generates a number of positional arguments
        to be passed during instantiation of the class

    init_native_arrays
        A search strategy that generates a boolean to test the method with native
        arrays

    init_as_variable_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a Variable array

    test_compile
        A search strategy that generates a boolean to graph compile and test the
        function

    precision_mode
        A search strategy that generates a boolean to switch between two different
        precision modes supported by numpy and (torch, jax) and test the function

    method_num_positional_args
        A search strategy that generates a number of positional arguments
        to be passed during call of the class method

    method_native_arrays
        A search strategy that generates a boolean to test the method with native
        arrays

    method_as_variable_flags
        A search strategy that generates a list of boolean flags for array inputs to be
        passed as a Variable array

    test_inplace
        A search strategy that generates a boolean to test the method with `inplace`
        update
    """
    split_index = init_tree.rfind(".")
    framework_init_module = init_tree[:split_index]
    ivy_init_module = f"ivy.functional.frontends.{init_tree[:split_index]}"
    init_name = init_tree[split_index + 1 :]
    init_tree = f"ivy.functional.frontends.{init_tree}"
    is_hypothesis_test = len(_given_kwargs) != 0

    split_index = class_tree.rfind(".")
    class_module_path, class_name = (
        class_tree[:split_index],
        class_tree[split_index + 1 :],
    )
    class_module = importlib.import_module(class_module_path)
    method_class = getattr(class_module, class_name)

    if is_hypothesis_test:
        callable_method = getattr(method_class, method_name)
        if init_num_positional_args is None:
            init_num_positional_args = num_positional_args(fn_name=init_tree)

        if method_num_positional_args is None:
            method_num_positional_args = num_positional_args_method(
                method=callable_method
            )

    def test_wrapper(test_fn):
        supported_device_dtypes = _get_method_supported_devices_dtypes(
            method_name, class_module, class_name
        )

        if is_hypothesis_test:
            param_names = inspect.signature(test_fn).parameters.keys()
            init_flags = pf.frontend_init_flags(
                num_positional_args=init_num_positional_args,
                as_variable=_get_runtime_flag_value(init_as_variable_flags),
                native_arrays=_get_runtime_flag_value(init_native_arrays),
            )

            method_flags = pf.frontend_method_flags(
                num_positional_args=method_num_positional_args,
                inplace=_get_runtime_flag_value(test_inplace),
                as_variable=_get_runtime_flag_value(method_as_variable_flags),
                native_arrays=_get_runtime_flag_value(method_native_arrays),
                test_trace=_get_runtime_flag_value(test_trace),
                test_trace_each=_get_runtime_flag_value(test_trace_each),
                precision_mode=_get_runtime_flag_value(precision_mode),
                generate_frontend_arrays=_get_runtime_flag_value(
                    generate_frontend_arrays
                ),
            )
            ivy_init_modules = str(ivy_init_module)
            framework_init_modules = str(framework_init_module)
            frontend_helper_data = FrontendMethodData(
                ivy_init_module=ivy_init_modules,
                framework_init_module=framework_init_modules,
                init_name=init_name,
                method_name=method_name,
            )

            possible_arguments = {
                "init_flags": init_flags,
                "method_flags": method_flags,
                "frontend_method_data": st.just(frontend_helper_data),
            }

            filtered_args = set(param_names).intersection(possible_arguments.keys())
            for key in filtered_args:
                # extend Hypothesis given kwargs with our strategies
                _given_kwargs[key] = possible_arguments[key]

            hypothesis_test_fn = given(**_given_kwargs)(test_fn)

            @functools.wraps(hypothesis_test_fn)
            def wrapped_test(*args, **kwargs):
                try:
                    hypothesis_test_fn(*args, **kwargs)
                except Exception as e:
                    # A string matching is used instead of actual exception due to
                    # exception object in with_backend is different from global Ivy
                    if e.__class__.__qualname__ == "IvyNotImplementedException":
                        pytest.skip("Function not implemented in backend.")
                    else:
                        raise e

        else:
            wrapped_test = test_fn

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            fn_tree=f"{init_tree}.{method_name}",
            fn_name=method_name,
            supported_device_dtypes=supported_device_dtypes,
            is_method=[method_name, class_tree, split_index],
        )

        return wrapped_test

    return test_wrapper


@st.composite
def seed(draw):
    return draw(st.integers(min_value=0, max_value=2**8 - 1))


def _create_transpile_report(
    data: dict, backend: str, file_name: str, is_backend: bool = False
):
    backend_specific_data = ["nodes", "time", "args", "kwargs"]
    # json report exists already
    if os.path.isfile(file_name):
        with open(file_name, "r") as outfile:
            # Load the file's existing data
            file_data = json.load(outfile)
            if file_data["nodes"].get(backend, 0) > data["nodes"]:
                return

            # that are backend specific
            for key in backend_specific_data:
                file_data[key][backend] = data[key]
            if not is_backend:
                # not backend specific
                for key in ["ivy_nodes", "fw_time"]:
                    file_data[key] = data[key]
            json_object = json.dumps(file_data, indent=6)
            with open(file_name, "w") as outfile:
                outfile.write(json_object)
            return
    # create new json report
    for key in backend_specific_data:
        data[key] = {backend: data[key]}
    json_object = json.dumps(data, indent=6)
    with open(file_name, "w") as outfile:
        outfile.write(json_object)


def handle_example(
    *,
    test_example: bool = False,
    test_frontend_example: bool = False,
    test_method_example: bool = False,
    test_frontend_method_example: bool = False,
    **given_kwargs,
):
    if test_example:
        test_flags = given_kwargs.get("test_flags", {})
        flags = pf.FunctionTestFlags(
            ground_truth_backend=test_flags.get("ground_truth_backend", "numpy"),
            num_positional_args=test_flags.get("num_positional_args", 0),
            instance_method=test_flags.get("instance_method", False),
            with_out=test_flags.get("with_out", False),
            with_copy=test_flags.get("with_copy", False),
            test_gradients=test_flags.get("test_gradients", False),
            test_trace=test_flags.get("test_trace", False),
            test_trace_each=test_flags.get("test_trace_each", False),
            transpile=test_flags.get("transpile", False),
            as_variable=test_flags.get("as_variable", [False]),
            native_arrays=test_flags.get("native_arrays", [False]),
            container=test_flags.get("container", [False]),
            precision_mode=test_flags.get("precision_mode", False),
            test_cython_wrapper=test_flags.get("test_cython_wrapper", False),
        )

        given_kwargs["test_flags"] = flags

    elif test_frontend_example:
        test_flags = given_kwargs.get("test_flags", {})
        flags = pf.FrontendFunctionTestFlags(
            num_positional_args=test_flags.get("num_positional_args", 0),
            with_out=test_flags.get("with_out", False),
            with_copy=test_flags.get("with_copy", False),
            inplace=test_flags.get("inplace", False),
            as_variable=test_flags.get("as_variable", [False]),
            native_arrays=test_flags.get("native_arrays", [False]),
            test_trace=test_flags.get("test_trace", False),
            test_trace_each=test_flags.get("test_trace_each", False),
            generate_frontend_arrays=test_flags.get("generate_frontend_arrays", False),
            transpile=test_flags.get("transpile", False),
            precision_mode=test_flags.get("precision_mode", False),
        )

        given_kwargs["test_flags"] = flags

    elif test_method_example:
        method_flags = given_kwargs.get("method_flags", {})
        init_flags = given_kwargs.get("init_flags", {})
        flags_1 = pf.MethodTestFlags(
            num_positional_args=method_flags.get("num_positional_args", 0),
            as_variable=method_flags.get("as_variable", [False]),
            native_arrays=method_flags.get("native_arrays", [False]),
            container_flags=method_flags.get("container", [False]),
            precision_mode=method_flags.get("precision_mode", False),
        )

        flags_2 = pf.InitMethodTestFlags(
            num_positional_args=init_flags.get("num_positional_args", 0),
            as_variable=init_flags.get("as_variable", [False]),
            native_arrays=init_flags.get("native_arrays", [False]),
            precision_mode=init_flags.get("precision_mode", False),
        )

        given_kwargs["method_flags"] = flags_1
        given_kwargs["init_flags"] = flags_2

    elif test_frontend_method_example:
        method_flags = given_kwargs.get("method_flags", {})
        init_flags = given_kwargs.get("init_flags", {})
        flags_1 = pf.FrontendMethodTestFlags(
            num_positional_args=method_flags.get("num_positional_args", 0),
            as_variable=method_flags.get("as_variable", [False]),
            native_arrays=method_flags.get("native_arrays", [False]),
            precision_mode=method_flags.get("precision_mode", False),
            inplace=method_flags.get("inplace", False),
            test_trace=method_flags.get("test_trace", False),
            test_trace_each=method_flags.get("test_trace_each", False),
            generate_frontend_arrays=method_flags.get(
                "generate_frontend_arrays", False
            ),
        )

        flags_2 = pf.FrontendInitTestFlags(
            num_positional_args=init_flags.get("num_positional_args", 0),
            as_variable=init_flags.get("as_variable", [False]),
            native_arrays=init_flags.get("native_arrays", [False]),
        )

        given_kwargs["method_flags"] = flags_1
        given_kwargs["init_flags"] = flags_2

    def test_wrapper(test_fn):

        hypothesis_test_fn = example(**given_kwargs)(test_fn)

        @functools.wraps(hypothesis_test_fn)
        def wrapped_test(*args, **kwargs):
            try:
                hypothesis_test_fn(*args, **kwargs)
            except Exception as e:
                # A string matching is used instead of actual exception due to
                # exception object in with_backend is different from global Ivy
                if e.__class__.__qualname__ == "IvyNotImplementedException":
                    pytest.skip("Function not implemented in backend.")
                else:
                    raise e

        return wrapped_test

    return test_wrapper
