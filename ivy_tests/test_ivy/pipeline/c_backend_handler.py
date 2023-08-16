import ivy
from ivy_tests.test_ivy.pipeline.base.backend_handler import BackendHandler


class SetBackendHandler(BackendHandler):
    def set_backend(self, backend: str):
        return ivy.set_backend(backend)

    def unset_backend(self):
        return ivy.unset_backend()


class WithBackendHandler(BackendHandler):
    def set_backend(self, backend: str):
        return ivy.with_backend(backend, cached=True)

    def unset_backend(self):
        return None
