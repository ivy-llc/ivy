from .. import config

if hasattr(config, "try_except"):
    config.try_except()
from . import helpers

test_shapes = ((), (1,), (2, 1), (1, 2, 3))
