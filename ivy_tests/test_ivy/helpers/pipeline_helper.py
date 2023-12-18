# TODO rename file
import ivy
import importlib
from typing import Callable


class WithBackendContext:
    def __init__(self, backend, cached=True) -> None:
        self.backend = backend
        self.cached = cached

    def __enter__(self):
        return ivy.with_backend(self.backend, cached=self.cached)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


# update_backend: Callable = ivy.utils.backend.ContextManager
update_backend: Callable = WithBackendContext


def get_frontend_config(frontend: str):
    return importlib.import_module(
        f"ivy_tests.test_ivy.test_frontends.config.{frontend}"
    )
