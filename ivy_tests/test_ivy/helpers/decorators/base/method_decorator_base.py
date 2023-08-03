import importlib
from ivy_tests.test_ivy.helpers.decorators.base.decorator_base import HandlerBase


class MethodHandlerBase(HandlerBase):
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
