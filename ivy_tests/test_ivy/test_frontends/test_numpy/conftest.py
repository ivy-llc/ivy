import pytest
from ...conftest import mod_backend, mod_frontend


@pytest.fixture(scope="session")
def frontend():
    if mod_frontend["numpy"]:
        return mod_frontend["numpy"]
    return "numpy"
