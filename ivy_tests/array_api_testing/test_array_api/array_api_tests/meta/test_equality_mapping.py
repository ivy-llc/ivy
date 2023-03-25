import pytest

from ..dtype_helpers import EqualityMapping


def test_raises_on_distinct_eq_key():
    with pytest.raises(ValueError):
        EqualityMapping([(float("nan"), "value")])


def test_raises_on_indistinct_eq_keys():
    class AlwaysEq:
        def __init__(self, hash):
            self._hash = hash

        def __eq__(self, other):
            return True

        def __hash__(self):
            return self._hash

    with pytest.raises(ValueError):
        EqualityMapping([(AlwaysEq(0), "value1"), (AlwaysEq(1), "value2")])


def test_key_error():
    mapping = EqualityMapping([("key", "value")])
    with pytest.raises(KeyError):
        mapping["nonexistent key"]


def test_iter():
    mapping = EqualityMapping([("key", "value")])
    it = iter(mapping)
    assert next(it) == "key"
    with pytest.raises(StopIteration):
        next(it)
