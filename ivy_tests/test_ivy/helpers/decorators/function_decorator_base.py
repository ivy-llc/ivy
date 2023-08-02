import importlib
import functools
import pytest
import ivy.functional.frontends.numpy as np_frontend  # TODO wtf?
import inspect

from abc import ABC, abstractmethod
from hypothesis import given
from ivy_tests.test_ivy.helpers.structs import ParametersInfo
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy_tests.test_ivy.helpers.pipeline_helper import update_backend
from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import (
    _dtype_kind_keys,
    _get_type_dict,
)
from typing import Callable, Any


class FunctionHandler(ABC):
    @abstractmethod
    def __init__(self, fn_tree: str, test_flags, **_given_kwargs):
        pass

    def _append_ivy_to_fn_tree(self, fn_tree):
        return "ivy." + fn_tree

    def _build_parameter_info(self, fn):
        total = num_positional_only = num_keyword_only = 0
        # TODO refactor out
        for param in inspect.signature(fn).parameters.values():
            if param.name == "self":
                continue
            total += 1
            if param.kind == param.POSITIONAL_ONLY:
                num_positional_only += 1
            elif param.kind == param.KEYWORD_ONLY:
                num_keyword_only += 1
            elif param.kind == param.VAR_KEYWORD:
                num_keyword_only += 1
        return ParametersInfo(
            total=total,
            positional_only=num_positional_only,
            keyword_only=num_keyword_only,
        )

    def _build_parameters_info_dict(self, fn_tree):
        ret = {}

        for framework in available_frameworks:
            with update_backend(framework) as ivy_backend:
                module_tree, fn_name = self._partition_fn_tree(fn_tree)
                module = ivy_backend.utils.dynamic_import.import_module(module_tree)
                fn = getattr(module, fn_name)
                parameter_info = self._build_parameter_info(fn)
                ret[framework] = parameter_info

        return ret

    def _get_supported_devices_dtypes(self, fn_tree: str):
        """
        Get supported devices and data types for a function in Ivy API.

        Parameters
        ----------
        fn_tree
            Full import path of the function module

        Returns
        -------
        Returns a dictonary containing supported device types and its supported data
        types for the function
        """
        supported_device_dtypes = {}

        fn_module, fn_name = self._partition_fn_tree(fn_tree)

        # This is for getting a private function from numpy frontend where we have
        # a ufunc object as we can't refer to them as functions
        if fn_tree.startswith("ivy.functional.frontends.numpy"):
            fn_module_ = np_frontend
            if isinstance(getattr(fn_module_, fn_name), fn_module_.ufunc):
                fn_name = "_" + fn_name

        for backend_str in available_frameworks:
            with update_backend(backend_str) as ivy_backend:
                module = ivy_backend.utils.dynamic_import.import_module(fn_module)
                callable_fn = module.__dict__[fn_name]
                # for partial mixed functions we should pass the backend function
                # to ivy.function_supported_devices_and_dtypes
                if (
                    hasattr(callable_fn, "mixed_backend_wrappers")
                    and ivy_backend.__dict__[fn_name] != callable_fn
                ):
                    callable_fn = ivy_backend.__dict__[fn_name]
                devices_and_dtypes = ivy_backend.function_supported_devices_and_dtypes(
                    callable_fn
                )
                devices_and_dtypes = (
                    tuple(devices_and_dtypes.values())
                    if "compositional" in devices_and_dtypes.keys()
                    else (devices_and_dtypes,)
                )
                # Issue with bfloat16 and tensorflow
                for device_and_dtype in devices_and_dtypes:
                    try:
                        if "bfloat16" in device_and_dtype["gpu"]:
                            tmp = list(device_and_dtype["gpu"])
                            tmp.remove("bfloat16")
                            device_and_dtype["gpu"] = tuple(tmp)
                    except KeyError:
                        pass
                organized_dtypes = {}
                all_organized_dtypes = []
                for device_and_dtype in devices_and_dtypes:
                    for device in device_and_dtype.keys():
                        organized_dtypes[device] = self._partition_dtypes_into_kinds(
                            backend_str, device_and_dtype[device]
                        )
                    all_organized_dtypes.append(organized_dtypes)
                supported_device_dtypes[backend_str] = (
                    {
                        "compositional": all_organized_dtypes[0],
                        "primary": all_organized_dtypes[1],
                    }
                    if len(all_organized_dtypes) > 1
                    else all_organized_dtypes[0]
                )
        return supported_device_dtypes

    def _partition_fn_tree(self, fn_tree: str):
        module_tree, _, fn_name = fn_tree.rpartition(".")
        return module_tree, fn_name

    def _partition_dtypes_into_kinds(self, framework: str, dtypes):
        partitioned_dtypes = {}
        for kind in _dtype_kind_keys:
            partitioned_dtypes[kind] = set(
                _get_type_dict(framework, kind)
            ).intersection(dtypes)
        return partitioned_dtypes

    def import_function(self, fn_tree: str) -> Callable[..., Any]:
        module_tree, _, fn_name = fn_tree.rpartition(".")
        module = importlib.import_module(module_tree)
        return getattr(module, fn_name)

    @property
    def is_hypothesis_test(self) -> bool:
        return len(self._given_kwargs.items()) > 0

    @abstractmethod
    def __call__(self, func: Callable[..., Any]):
        pass

    def _wrap_with_hypothesis(self, func: Callable[..., Any]):
        return given(**self._given_kwargs)(func)

    def _handle_not_implemented(self, func):
        @functools.wraps(func)
        def wrapped_test(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                # A string matching is used instead of actual exception due to
                # exception object in with_backend is different from global Ivy
                if e.__class__.__qualname__ == "IvyNotImplementedException":
                    pytest.skip("Function not implemented in backend.")
                else:
                    raise e

        return wrapped_test
