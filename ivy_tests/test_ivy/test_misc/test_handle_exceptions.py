import hypothesis.strategies as st
import pytest
from hypothesis import given

from ivy import handle_exceptions
from ivy.utils.exceptions import _non_ivy_exceptions_mapping
from ivy.utils.exceptions import InplaceUpdateException
from ivy.utils.exceptions import IvyAttributeError
from ivy.utils.exceptions import IvyBackendException
from ivy.utils.exceptions import IvyBroadcastShapeError
from ivy.utils.exceptions import IvyDeviceError
from ivy.utils.exceptions import IvyDtypePromotionError
from ivy.utils.exceptions import IvyError
from ivy.utils.exceptions import IvyException
from ivy.utils.exceptions import IvyIndexError
from ivy.utils.exceptions import IvyInvalidBackendException
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.utils.exceptions import IvyValueError


@handle_exceptions
def func(e):
    if e is None:
        return

    raise e()


@pytest.mark.parametrize(
    "e",
    (
        IvyError,
        IvyNotImplementedException,
        IvyBroadcastShapeError,
        IvyValueError,
        InplaceUpdateException,
        IvyException,
        IvyIndexError,
        IvyAttributeError,
        IvyBackendException,
        IvyDeviceError,
        IvyInvalidBackendException,
        IvyDtypePromotionError,
    ),
)
def test_ivy_errors_raising(e):
    with pytest.raises(e):
        func(e)


def test_no_exception():
    func(None)


@pytest.mark.parametrize(
    "e, to_be_raised",
    _non_ivy_exceptions_mapping.items(),
)
def test_non_ivy_errors_mapping(e, to_be_raised):
    with pytest.raises(
        to_be_raised,
    ) as raised:
        func(e)
    assert issubclass(raised.type, to_be_raised)


@given(
    e=st.sampled_from(
        [
            Exception,
            ZeroDivisionError,
            BufferError,
            AssertionError,
            ImportError,
            KeyError,
            LookupError,
        ]
    )
)
def test_non_ivy_errors_raising(e):
    with pytest.raises(IvyBackendException):
        func(e)
