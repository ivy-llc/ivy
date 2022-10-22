import pytest


@pytest.fixture(scope="session")
def fixt_frontend_str():
    return "jax"
