import hypothesis
import pytest
import ivy
from hypothesis import strategies as st, given, settings
import itertools

backends = ["numpy", "jax", "tensorflow", "torch"]


@pytest.fixture
def compiled_backends():
    compiled_backends = []
    print(ivy.backend, id(ivy))
    for b in backends:
        _b = ivy.utils.backend.compiler.with_backend(b)
        compiled_backends.append(_b)
    return compiled_backends


@settings(
    # To be able to share compiled_backends between examples
    suppress_health_check=[hypothesis.HealthCheck(9)]
)
@given(name=st.sampled_from(["add", "Array", "Container", "globals"]))
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
