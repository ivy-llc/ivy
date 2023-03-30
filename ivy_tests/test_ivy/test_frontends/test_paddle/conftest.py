import pytest
from ...conftest import mod_frontend


@pytest.fixture(scope="session")
def frontend():
    if mod_frontend["paddle"]:
        return mod_frontend["paddle"]
    return "paddle"
