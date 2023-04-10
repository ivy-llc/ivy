try:
    from jax.config import config

    config.update("jax_enable_x64", True)
except (ImportError, RuntimeError):
    pass
