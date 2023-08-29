from .base import FrontendConfigWithBackend


class JaxFrontendConfig(FrontendConfigWithBackend):
    backend_str = "jax"


def get_config():
    return JaxFrontendConfig()
