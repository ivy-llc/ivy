import importlib
from abc import ABC, abstractmethod, abstractproperty
from hypothesis import given
from typing import List, Callable, Any


class FunctionHandler(ABC):
    @abstractmethod
    def __init__(self, fn_tree: str, test_flags, **_given_kwargs):
        pass

    def import_function(self, fn_tree: str) -> Callable[..., Any]:
        module_tree, _, fn_name = fn_tree.rpartition(".")
        module = importlib.import_module(module_tree)
        return getattr(module, fn_name)

    @abstractproperty
    def is_hypothesis_test(self) -> List[str]:
        pass

    @abstractproperty
    def parameters(self) -> List[str]:
        pass

    @abstractmethod
    def __call__(self, func: Callable[..., Any]):
        pass

    def _wrap_with_hypothesis(self, func: Callable[..., Any]):
        return given(test_flags=self.test_flags, **self._given_kwargs)(func)
