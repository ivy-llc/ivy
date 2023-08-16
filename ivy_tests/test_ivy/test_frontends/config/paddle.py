from .base import FrontendConfigWithBackend


def get_config():
    return PaddleFrontendConfig()


class PaddleFrontendConfig(FrontendConfigWithBackend):
    backend_str = "paddle"
