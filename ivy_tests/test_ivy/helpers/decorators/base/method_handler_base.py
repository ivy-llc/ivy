from abc import abstractproperty
import ivy_tests.test_ivy.helpers.decorators.import_helpers as import_helpers
from ivy_tests.test_ivy.helpers.decorators.base.handler_base import HandlerBase
from ivy_tests.test_ivy.helpers.pipeline_helper import update_backend
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks


class MethodHandlerBase(HandlerBase):
    @abstractproperty
    def init_tree():
        pass

    @abstractproperty
    def method_tree():
        pass

    def _build_supported_devices_dtypes(self):
        supported_device_dtypes = {}
        module_tree, class_name, method_name = import_helpers.partition_method_tree(
            self.method_tree
        )
        for backend_str in available_frameworks:
            with update_backend(backend_str) as backend:
                # TODO ineffecient due to calling context manager twice
                method = import_helpers.import_method_from_ivy(
                    module_tree, class_name, method_name, backend_str
                )
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
