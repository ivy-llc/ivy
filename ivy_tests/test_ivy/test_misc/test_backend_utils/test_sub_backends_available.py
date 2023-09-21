from importlib.util import find_spec

import pytest

import ivy


def test_no_warning_when_no_sub_backend_implementation_available():
    ivy.set_backend("numpy")
    q = ivy.array([[[0.2, 1.0], [2.2, 3.0], [4.4, 5.6]]])
    k = ivy.array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
    v = ivy.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
    with pytest.warns(None) as record:
        ivy.scaled_dot_product_attention(
            q, k, v, scale=1, dropout_p=0.1, is_causal=True, training=True
        )
    assert len(record) == 0


@pytest.mark.skipif(find_spec("xformers") is None, reason="xformers is not installed")
def test_sub_backend_implementation_available():
    ivy.set_backend("torch")
    sub_backends = ivy.available_sub_backend_implementations(
        "scaled_dot_product_attention"
    )
    assert "xformers" in sub_backends


def test_sub_backend_implementation_not_available():
    ivy.set_backend("numpy")
    sub_backends = ivy.available_sub_backend_implementations(
        "scaled_dot_product_attention"
    )
    assert not sub_backends


@pytest.mark.skipif(find_spec("xformers") is None, reason="xformers is not installed")
def test_throw_warning_when_sub_backend_implementation_available_but_not_used():
    ivy.set_backend("torch")
    q = ivy.array([[[0.2, 1.0], [2.2, 3.0], [4.4, 5.6]]])
    k = ivy.array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
    v = ivy.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
    with pytest.warns(UserWarning):
        ivy.scaled_dot_product_attention(
            q, k, v, scale=1, dropout_p=0.1, is_causal=True, training=True
        )
