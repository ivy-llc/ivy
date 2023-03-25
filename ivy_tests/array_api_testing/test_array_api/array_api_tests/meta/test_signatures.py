from inspect import Parameter, Signature, signature

import pytest

from ..test_signatures import _test_inspectable_func


def stub(foo, /, bar=None, *, baz=None):
    pass


stub_sig = signature(stub)


@pytest.mark.parametrize(
    "sig",
    [
        Signature(
            [
                Parameter("foo", Parameter.POSITIONAL_ONLY),
                Parameter("bar", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("baz", Parameter.KEYWORD_ONLY),
            ]
        ),
        Signature(
            [
                Parameter("foo", Parameter.POSITIONAL_ONLY),
                Parameter("bar", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("baz", Parameter.POSITIONAL_OR_KEYWORD),
            ]
        ),
        Signature(
            [
                Parameter("foo", Parameter.POSITIONAL_ONLY),
                Parameter("bar", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("qux", Parameter.KEYWORD_ONLY),
                Parameter("baz", Parameter.KEYWORD_ONLY),
            ]
        ),
    ],
)
def test_good_sig_passes(sig):
    _test_inspectable_func(sig, stub_sig)


@pytest.mark.parametrize(
    "sig",
    [
        Signature(
            [
                Parameter("foo", Parameter.POSITIONAL_ONLY),
                Parameter("bar", Parameter.POSITIONAL_ONLY),
                Parameter("baz", Parameter.KEYWORD_ONLY),
            ]
        ),
        Signature(
            [
                Parameter("foo", Parameter.POSITIONAL_ONLY),
                Parameter("bar", Parameter.KEYWORD_ONLY),
                Parameter("baz", Parameter.KEYWORD_ONLY),
            ]
        ),
    ],
)
def test_raises_on_bad_sig(sig):
    with pytest.raises(AssertionError):
        _test_inspectable_func(sig, stub_sig)
