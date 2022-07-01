from hypothesis import settings, HealthCheck
from pytest import mark

settings.register_profile("ci", suppress_health_check=(HealthCheck(3),))
settings.load_profile("ci")

skip_ids = []

skips_path = "ivy_tests/skips.txt"
if skips_path.exists():
    with open(skips_path) as f:
        for line in f:
            if line.startswith("array_api_tests"):
                id_ = line.strip("\n")
                skip_ids.append(id_)


def pytest_collection_modifyitems(config, items):
    for item in items:
        # skip if specified in skips.txt
        for id_ in skip_ids:
            if item.nodeid.startswith(id_):
                item.add_marker(
                    mark.skip(
                        reason="failed health check-too much data \
                        filtered in hypothesis test"
                    )
                )
                break
