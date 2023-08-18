# TODO rename file
from enum import Enum
import ivy
import importlib


class BackendHandlerMode(Enum):
    WithBackend = 0
    SetBackend = 1


class WithBackendContext:
    def __init__(self, backend) -> None:
        self.backend = backend

    def __enter__(self):
        return ivy.with_backend(self.backend)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class BackendHandler:
    _context = WithBackendContext

    @classmethod
    def _update_context(cls, mode: BackendHandlerMode):
        if mode == BackendHandlerMode.WithBackend:
            cls._context = WithBackendContext
        elif mode == BackendHandlerMode.SetBackend:
            cls._context = ivy.utils.backend.ContextManager
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
