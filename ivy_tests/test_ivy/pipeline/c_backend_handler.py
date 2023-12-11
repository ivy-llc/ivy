import ivy
from ivy_tests.test_ivy.pipeline.base.backend_handler import BackendHandler


class SetBackendHandler(BackendHandler):
    def set_backend(self, backend: str):
        self._backend = backend
        return ivy.set_backend(backend)

    def unset_backend(self):
        self._backend = "None"
        return ivy.unset_backend()


class WithBackendHandler(BackendHandler):
    def set_backend(self, backend: str):
        self._backend = backend
        return ivy.with_backend(backend, cached=True)

    def unset_backend(self):
        self._backend = "None"
        return None
