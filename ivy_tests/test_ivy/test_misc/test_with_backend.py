# Global
import pytest
import itertools
from hypothesis import strategies as st, given, settings, HealthCheck

# Local
import ivy
import numpy as np
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


def test_prevent_access(backend_fw):
    local_ivy = ivy.with_backend(backend_fw)
    with pytest.raises(RuntimeError):
        local_ivy.with_backend(backend_fw)

    with pytest.raises(RuntimeError):
        local_ivy.set_backend(backend_fw)


def test_with_backend_cached(backend_fw):
    non_cached_local_ivy = ivy.with_backend(backend_fw)
    cached_local_ivy = ivy.with_backend(backend_fw, cached=True)
    assert non_cached_local_ivy == cached_local_ivy


def test_is_local(backend_fw):
    local_ivy = ivy.with_backend(backend_fw, cached=True)
    assert local_ivy.is_local()


def test_with_backend_array(backend_fw):
    local_ivy = ivy.with_backend(backend_fw, cached=True)
    local_x = local_ivy.array([1, 2, 3, 4])
    ivy.set_backend(backend_fw)
    x = ivy.array([1, 2, 3, 4])
    assert np.allclose(x._data, local_x._data)
