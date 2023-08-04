import importlib
import ivy.functional.frontends.numpy as np_frontend  # TODO wtf?
import inspect

from abc import abstractproperty
from ivy_tests.test_ivy.helpers.structs import ParametersInfo
from ivy_tests.test_ivy.helpers.globals import TestData
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy_tests.test_ivy.helpers.pipeline_helper import update_backend
from ivy_tests.test_ivy.helpers.decorators.base.decorator_base import HandlerBase
from ivy_tests.test_ivy.helpers.decorators.strategies import (
    num_positional_args_from_dict,
)
from typing import Callable, Any


class FunctionHandler(HandlerBase):
    @abstractproperty
    def possible_args():
        pass

    def __call__(self, fn: Callable[..., Any]):
        if self.is_hypothesis_test:
            self._update_given_kwargs(fn)
            wrapped_fn = self._wrap_with_hypothesis(fn)
        else:
            wrapped_fn = fn

        self._add_test_attributes_to_test_function(wrapped_fn)
        self._handle_not_implemented(wrapped_fn)
        return wrapped_fn

    def _build_test_data(self):
        module_tree, fn_name = self._partition_fn_tree(self.fn_tree)
        supported_device_dtypes = self._get_supported_devices_dtypes(self.fn_tree)
        self.test_data = TestData(
            module_tree=module_tree,
            fn_name=fn_name,
            supported_device_dtypes=supported_device_dtypes,
        )

    def _update_given_kwargs(self, fn):
        param_names = inspect.signature(fn).parameters.keys()

        # Check if these arguments are being asked for
        filtered_args = set(param_names).intersection(self.possible_args.keys())
        for key in filtered_args:
            self._given_kwargs[key] = self.possible_args[key]

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

    def _build_num_positional_arguments_strategy(self):
        dict_for_num_pos_strategy = self._build_parameters_info_dict(self.fn_tree)
        return num_positional_args_from_dict(dict_for_num_pos_strategy)

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

    def import_function(self, fn_tree: str) -> Callable[..., Any]:
        module_tree, _, fn_name = fn_tree.rpartition(".")
        module = importlib.import_module(module_tree)
        return getattr(module, fn_name)
