# global
import pytest

# local
from ivy import unset_backend


@pytest.fixture(autouse=True)
def run_around_tests():
    unset_backend()
