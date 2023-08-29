from .base import FrontendConfigWithBackend


class TorchFrontendConfig(FrontendConfigWithBackend):
    backend_str = "torch"


def get_config():
    return TorchFrontendConfig()
