from hypothesis import settings, HealthCheck
from pytest import mark
from os import path

# flake8: noqa
settings.register_profile("ci", suppress_health_check=(HealthCheck(3),))
settings.load_profile("ci")

skip_ids = [
    "ivy/ivy_tests/test_array_api/array_api_tests/test_array_object.py::test_getitem_masking",
    "ivy/ivy_tests/test_array_api/array_api_tests/test_array_object.py::test_setitem",
]

skips_path = r"C:\ivy_tests\skips.txt"
if path.exists("skips.txt"):
    with open("skips.txt") as f:
        for line in f:
            print("skip:", line)
            # if line.startswith("test_array_api"):
            id_ = line.strip("\n")
            skip_ids.append(id_)


def pytest_collection_modifyitems(items):
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
