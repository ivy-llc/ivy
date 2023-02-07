import pytest
from ...conftest import mod_backend, mod_frontend


@pytest.fixture(scope="session")
def frontend():
    if mod_frontend["torch"]:
        return mod_frontend["torch"]
    elif mod_backend["torch"]:
        return mod_backend["torch"]
    return "torch"
