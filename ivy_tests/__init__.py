from jax.config import config
config.update("jax_enable_x64", True)

from hypothesis import settings, HealthCheck

settings.register_profile("ci", suppress_health_check=(HealthCheck(3),))
settings.load_profile("ci")
