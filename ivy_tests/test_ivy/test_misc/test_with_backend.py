# Global
import pytest
import itertools
from hypothesis import strategies as st, given, settings, HealthCheck

# Local
import ivy
from ivy.utils.backend.handler import _backend_dict


@pytest.fixture
def compiled_backends():
    compiled_backends = []
    for b in _backend_dict:
        _b = ivy.with_backend(b)
        compiled_backends.append(_b)
    return compiled_backends


@settings(
    # To be able to share compiled_backends between examples
    suppress_health_check=[HealthCheck(9)]
)
@given(name=st.sampled_from(["add", "Array", "Container", "globals_vars"]))
def test_memory_id(name, compiled_backends):
    for b in compiled_backends:
        assert id(getattr(ivy, name)) != id(
            getattr(b, name)
        ), f"Shared object {name} between global Ivy and backend {b.backend}"

    for comb in itertools.combinations(compiled_backends, 2):
        assert id(getattr(comb[0], name)) != id(getattr(comb[1], name)), (
            f"Shared object {name} between {comb[0].backend} and backend "
            f"{comb[1].backend}"
        )
