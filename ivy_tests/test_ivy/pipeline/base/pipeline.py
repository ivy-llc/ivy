from abc import ABC, abstractmethod
from ivy_tests.test_ivy.pipeline.c_backend_handler import (
    WithBackendHandler,
    SetBackendHandler,
)


class Pipeline(ABC):
    traced_fn = None
    backend_handler = WithBackendHandler()

    @classmethod
    def set_with_backend_handler(cls):
        cls.backend_handler = WithBackendHandler()

    @classmethod
    def set_set_backend_handler(cls):
        cls.backend_handler = SetBackendHandler()

    @classmethod
    def set_traced_fn(cls, fn):
        cls.traced_fn = fn

    @abstractmethod
    def test_function(self):
        pass

    @abstractmethod
    def test_method(self):
        pass
