import pytest
from ...conftest import mod_frontend


@pytest.fixture(scope="session")
def frontend():
    if mod_frontend["jax"]:
        return mod_frontend["jax"]
    return "jax"
