# TODO rename file
from enum import Enum
from typing import Callable
import ivy
import importlib


class BackendHandlerMode(Enum):
    WithBackend = 0
    SetBackend = 1


class WithBackendContext:
    def __init__(self, backend, cached=True) -> None:
        self.backend = backend
        self.cached = cached

    def __enter__(self):
        return ivy.with_backend(self.backend, cached=self.cached)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


update_backend: Callable = ivy.utils.backend.ContextManager


# update_backend: Callable = WithBackendContext
class BackendHandler:
    _context = WithBackendContext
    _ctx_flag = 0  # BackendHandlerMode configs

    @classmethod
    def _update_context(cls, mode: BackendHandlerMode):
        if mode == BackendHandlerMode.WithBackend:
            cls._context = WithBackendContext
            cls._ctx_flag = 0
        elif mode == BackendHandlerMode.SetBackend:
            cls._context = ivy.utils.backend.ContextManager
            cls._ctx_flag = 1
        else:
            raise ValueError(f"Unknown backend handler mode! {mode}")

    @classmethod
    def update_backend(cls, backend):
        return cls._context(backend)


def get_frontend_config(frontend: str):
    config_module = importlib.import_module(
        f"ivy_tests.test_ivy.test_frontends.config.{frontend}"
    )
    return config_module.get_config()
