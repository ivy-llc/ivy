from abc import ABC, abstractmethod
from ivy_tests.test_ivy.pipeline.c_backend_handler import (
    WithBackendHandler,
    SetBackendHandler,
)


class Pipeline(ABC):
    mod_backend = {
        "numpy": None,
        "jax": None,
        "tensorflow": None,
        "torch": None,
        "paddle": None,
        "mxnet": None,
    }
    multiprocessing_flag = False
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

    @classmethod
    def set_mod_backend(cls, mod_backend):
        for key in mod_backend:
            cls.mod_backend[key] = mod_backend[key]
        cls.multiprocessing_flag = True
        print("mod backenddddddd")
        print(cls.mod_backend)

    @abstractmethod
    def test_function(self):
        pass

    @abstractmethod
    def test_method(self):
        pass
