from .base import FrontendConfigWithBackend


def get_config():
    return TensorflowFrontendConfig()


class TensorflowFrontendConfig(FrontendConfigWithBackend):
    backend_str = "tensorflow"
