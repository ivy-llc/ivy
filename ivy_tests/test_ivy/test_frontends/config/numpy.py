from .base import FrontendConfigWithBackend


class NumpyFrontendConfig(FrontendConfigWithBackend):
    backend_str = "numpy"


def get_config():
    return NumpyFrontendConfig()
