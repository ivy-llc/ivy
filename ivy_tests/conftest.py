import os
import redis
from hypothesis import settings, HealthCheck
from hypothesis.extra.redis import RedisExampleDatabase
from pytest import mark
from pathlib import Path

if "REDIS_CONNECTION_URL" in os.environ:
    r = redis.Redis.from_url(
        os.environ["REDIS_URL"], password=os.environ["REDIS_PASSWD"]
    )
    settings.register_profile(
        "ci_with_db",
        database=RedisExampleDatabase(r, key_prefix=b"hypothesis-example:"),
        suppress_health_check=(HealthCheck(3), HealthCheck(2)),
    )
    settings.load_profile("ci_with_db")

else:
    settings.register_profile(
        "ci", suppress_health_check=(HealthCheck(3), HealthCheck(2))
    )
    settings.load_profile("ci")


skip_ids = []
skips_path = Path(__file__).parent / "skips.txt"
if skips_path.exists():
    with open(skips_path) as f:
        for line in f:
            if line.startswith("ivy_tests"):
                id_ = line.strip("\n")
                skip_ids.append(id_)


def pytest_collection_modifyitems(items):
    skip_ivy = mark.skip(reason="ivy skip - see ivy_tests/skips.txt for details")
    for item in items:
        # skip if specified in skips.txt
        for id_ in skip_ids:
            if item.nodeid.startswith(id_):
                item.add_marker(skip_ivy)
                break
