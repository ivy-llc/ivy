import importlib

from abc import abstractproperty
from ivy_tests.test_ivy.helpers.decorators.base.decorator_base import HandlerBase
from ivy_tests.test_ivy.helpers.pipeline_helper import update_backend
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks


class MethodHandlerBase(HandlerBase):
    @abstractproperty
    def init_tree():
        pass

    @abstractproperty
    def method_tree():
        pass

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
