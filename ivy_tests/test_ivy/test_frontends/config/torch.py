from .base import FrontendConfigWithBackend


def get_config():
    return TorchFrontendConfig()


class TorchFrontendConfig(FrontendConfigWithBackend):
    backend_str = "torch"
