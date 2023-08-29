from .base import FrontendConfigWithBackend


class TensorflowFrontendConfig(FrontendConfigWithBackend):
    backend_str = "tensorflow"


def get_config():
    return TensorflowFrontendConfig()
