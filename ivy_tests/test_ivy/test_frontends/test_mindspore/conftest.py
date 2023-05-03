import pytest
from ...conftest import mod_frontend


@pytest.fixture(scope="session")
def frontend():
    if mod_frontend["mindspore"]:
        return mod_frontend["mindspore"]
    return "mindspore"
