import pytest
from ...conftest import mod_backend, mod_frontend


@pytest.fixture(scope="session")
def frontend():
    if mod_frontend["tensorflow"]:
        return mod_frontend["tensorflow"]
    elif mod_backend["tensorflow"]:
        return mod_backend["tensorflow"]
    return "tensorflow"
