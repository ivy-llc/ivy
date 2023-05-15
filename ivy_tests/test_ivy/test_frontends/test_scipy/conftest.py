import pytest
from ...conftest import mod_frontend


@pytest.fixture(scope="session")
def frontend():
    if mod_frontend["scipy"]:
        return mod_frontend["scipy"]
    return "scipy"
