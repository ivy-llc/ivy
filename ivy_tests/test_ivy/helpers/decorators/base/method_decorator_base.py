import inspect
import importlib
from typing import Any, Callable
from ivy_tests.test_ivy.helpers.structs import ParametersInfo

from abc import abstractproperty
from ivy_tests.test_ivy.helpers.decorators.base.decorator_base import HandlerBase
from ivy_tests.test_ivy.helpers.pipeline_helper import update_backend
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy_tests.test_ivy.helpers.decorators.strategies import (
    num_positional_args_from_dict,
)


class MethodHandlerBase(HandlerBase):
    @abstractproperty
    def init_tree():
        pass

    @abstractproperty
    def method_tree():
        pass

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

    def _build_parameters_info_dict(self, method_tree):
        ret = {}

        for framework in available_frameworks:
            with update_backend(framework) as ivy_backend:
                method = self._import_method(method_tree)
                parameter_info = self._build_parameter_info(method)
                ret[framework] = parameter_info

        return ret

    def _build_num_positional_arguments_strategy(self, method_tree: str):
        dict_for_num_pos_strategy = self._build_parameters_info_dict(method_tree)
        return num_positional_args_from_dict(dict_for_num_pos_strategy)

    def _partition_method_tree(self, method_tree: str):
        class_module_and_name, _, method_name = method_tree.rpartition(".")
        class_module, _, class_name = class_module_and_name.rpartition(".")
        return class_module, class_name, method_name

    def _import_method(self, method_tree: str):
        class_module, class_name, method_name = self._partition_method_tree(method_tree)
        module = importlib.import_module(class_module)
        cls = getattr(module, class_name)
        method = getattr(cls, method_name)
        return method

    def _build_supported_devices_dtypes(self):
        supported_device_dtypes = {}
        for backend_str in available_frameworks:
            with update_backend(backend_str) as backend:
                method = self._import_method(self.method_tree)
                devices_and_dtypes = backend.function_supported_devices_and_dtypes(
                    method
                )
                organized_dtypes = {}
                for device in devices_and_dtypes.keys():
                    organized_dtypes[device] = self._partition_dtypes_into_kinds(
                        backend_str, devices_and_dtypes[device]
                    )
                supported_device_dtypes[backend_str] = organized_dtypes
        return supported_device_dtypes

    def __call__(self, fn: Callable[..., Any]):
        if self.is_hypothesis_test:
            self._update_given_kwargs(fn)
            wrapped_fn = self._wrap_with_hypothesis(fn)
        else:
            wrapped_fn = fn

        self._add_test_attributes_to_test_function(wrapped_fn)
        self._handle_not_implemented(wrapped_fn)
        return wrapped_fn
