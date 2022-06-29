from hypothesis import settings, HealthCheck

settings.register_profile("ci", suppress_health_check=(HealthCheck(3), HealthCheck(2)))
settings.load_profile("ci")
