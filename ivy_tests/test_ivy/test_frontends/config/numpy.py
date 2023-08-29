from .base import FrontendConfigWithBackend


def get_config():
    return NumpyFrontendConfig()


class NumpyFrontendConfig(FrontendConfigWithBackend):
    backend_str = "numpy"
