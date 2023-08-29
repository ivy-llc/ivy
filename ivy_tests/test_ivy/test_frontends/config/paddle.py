from .base import FrontendConfigWithBackend


class PaddleFrontendConfig(FrontendConfigWithBackend):
    backend_str = "paddle"


def get_config():
    return PaddleFrontendConfig()
