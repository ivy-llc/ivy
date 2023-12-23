from .base import FrontendConfigWithBackend


def get_config():
    return JaxFrontendConfig()


class JaxFrontendConfig(FrontendConfigWithBackend):
    backend_str = "jax"
