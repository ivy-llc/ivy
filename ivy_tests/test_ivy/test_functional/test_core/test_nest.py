"""Collection of tests for unified general functions."""

# global
import copy
import warnings
import pytest
import numpy as np

# local
import ivy

# Helpers #
# --------#


def _snai(n, idx, v):
    if len(idx) == 1:
        n[idx[0]] = v
    else:
        _snai(n[idx[0]], idx[1:], v)


def _mnai(n, idx, fn):
    if len(idx) == 1:
        n[idx[0]] = fn(n[idx[0]])
    else:
        _mnai(n[idx[0]], idx[1:], fn)


def _pnai(n, idx):
    if len(idx) == 1:
        del n[idx[0]]
    else:
        _pnai(n[idx[0]], idx[1:])


# only checking for dicts but can test other nested functions using
# collections.abc.Sequences/Mapping/Iterable
def apply_fn_to_list(item, fun):
    if isinstance(item, list):
        return [apply_fn_to_list(x, fun) for x in item]
    else:
        return fun(item)


def map_nested_dicts(ob, func):
    for k, v in ob.items():
        if isinstance(v, dict):
            map_nested_dicts(v, func)
        else:
            ob[k] = apply_fn_to_list(v, func)


# Tests #
# ------#


# index_nest
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": (((2,), (4,)), ((6,), (8,)))}}]
)
@pytest.mark.parametrize(
    "index", [("a", 0, 0), ("a", 1, 0), ("b", "c", 0), ("b", "c", 1, 0)]
)
def test_index_nest(nest, index):
    ret = ivy.index_nest(nest, index)
    true_ret = nest
    for i in index:
        true_ret = true_ret[i]
    assert ret == true_ret


# set_nest_at_index
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize(
    "index", [("a", 0, 0), ("a", 1, 0), ("b", "c", 0), ("b", "c", 1, 0)]
)
@pytest.mark.parametrize("value", [-1])
@pytest.mark.parametrize("shallow", [True, False])
def test_set_nest_at_index(nest, index, value, shallow):
    nest_copy = copy.deepcopy(nest)
    result = ivy.set_nest_at_index(nest, index, value, shallow=shallow)
    _snai(nest_copy, index, value)

    assert result == nest_copy
    if shallow:
        assert nest == nest_copy
    else:
        assert nest != nest_copy


# map_nest_at_index
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize(
    "index", [("a", 0, 0), ("a", 1, 0), ("b", "c", 0, 0, 0), ("b", "c", 1, 0, 0)]
)
@pytest.mark.parametrize("fn", [lambda x: x + 2])
@pytest.mark.parametrize("shallow", [True, False])
def test_map_nest_at_index(nest, index, fn, shallow):
    nest_copy = copy.deepcopy(nest)
    result = ivy.map_nest_at_index(nest, index, fn, shallow=shallow)
    _mnai(nest_copy, index, fn)

    assert result == nest_copy
    if shallow:
        assert nest == nest_copy
    else:
        assert nest != nest_copy


# multi_index_nest
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": (((2,), (4,)), ((6,), (8,)))}}]
)
@pytest.mark.parametrize(
    "multi_indices", [(("a", 0, 0), ("a", 1, 0)), (("b", "c", 0), ("b", "c", 1, 0))]
)
def test_multi_index_nest(nest, multi_indices):
    rets = ivy.multi_index_nest(nest, multi_indices)
    true_rets = list()
    for indices in multi_indices:
        true_ret = nest
        for i in indices:
            true_ret = true_ret[i]
        true_rets.append(true_ret)
    assert rets == true_rets


# set_nest_at_indices
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize(
    "indices", [(("a", 0, 0), ("a", 1, 0)), (("b", "c", 0), ("b", "c", 1, 0))]
)
@pytest.mark.parametrize("values", [(1, 2)])
@pytest.mark.parametrize("shallow", [False, True])
def test_set_nest_at_indices(nest, indices, values, shallow):
    nest_copy = copy.deepcopy(nest)
    result = ivy.set_nest_at_indices(nest, indices, values, shallow=shallow)

    def snais(n, idxs, vs):
        [_snai(n, index, value) for index, value in zip(idxs, vs)]

    snais(nest_copy, indices, values)

    assert result == nest_copy
    if shallow:
        assert nest == nest_copy
    else:
        assert nest != nest_copy


# map_nest_at_indices
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize(
    "indices", [(("a", 0, 0), ("a", 1, 0)), (("b", "c", 0, 0, 0), ("b", "c", 1, 0, 0))]
)
@pytest.mark.parametrize("fn", [lambda x: x + 2, lambda x: x**2])
@pytest.mark.parametrize("shallow", [True, False])
def test_map_nest_at_indices(nest, indices, fn, shallow):
    nest_copy = copy.deepcopy(nest)
    result = ivy.map_nest_at_indices(nest, indices, fn, shallow)

    def mnais(n, idxs, vs):
        [_mnai(n, index, vs) for index in idxs]

    mnais(nest_copy, indices, fn)

    assert result == nest_copy
    if shallow:
        assert nest == nest_copy
    else:
        assert nest != nest_copy


# nested_argwhere
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
def test_nested_argwhere(nest):
    indices = ivy.nested_argwhere(nest, lambda x: x < 5)
    assert indices[0] == ["a", 0, 0]
    assert indices[1] == ["a", 1, 0]
    assert indices[2] == ["b", "c", 0, 0, 0]
    assert indices[3] == ["b", "c", 0, 1, 0]


# nested_argwhere_w_nest_checks
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
def test_nested_argwhere_w_nest_checks(nest):
    indices = ivy.nested_argwhere(
        nest, lambda x: isinstance(x, list) or (isinstance(x, int) and x < 5), True
    )
    assert indices[0] == ["a", 0, 0]
    assert indices[1] == ["a", 0]
    assert indices[2] == ["a", 1, 0]
    assert indices[3] == ["a", 1]
    assert indices[4] == ["a"]
    assert indices[5] == ["b", "c", 0, 0, 0]
    assert indices[6] == ["b", "c", 0, 0]
    assert indices[7] == ["b", "c", 0, 1, 0]
    assert indices[8] == ["b", "c", 0, 1]
    assert indices[9] == ["b", "c", 0]
    assert indices[10] == ["b", "c", 1, 0]
    assert indices[11] == ["b", "c", 1, 1]
    assert indices[12] == ["b", "c", 1]
    assert indices[13] == ["b", "c"]


# nested_argwhere_w_extra_nest_types
def test_nested_argwhere_w_extra_nest_types():
    nest = {"a": ivy.array([[0], [1]]), "b": {"c": ivy.array([[[2], [4]], [[6], [8]]])}}
    indices = ivy.nested_argwhere(nest, lambda x: x < 5, extra_nest_types=ivy.Array)
    assert indices[0] == ["a", 0, 0]
    assert indices[1] == ["a", 1, 0]
    assert indices[2] == ["b", "c", 0, 0, 0]
    assert indices[3] == ["b", "c", 0, 1, 0]


# all_nested_indices
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
def test_all_nested_indices(nest):
    indices = ivy.all_nested_indices(nest)
    assert indices[0] == ["a", 0, 0]
    assert indices[1] == ["a", 1, 0]
    assert indices[2] == ["b", "c", 0, 0, 0]
    assert indices[3] == ["b", "c", 0, 1, 0]
    assert indices[4] == ["b", "c", 1, 0, 0]
    assert indices[5] == ["b", "c", 1, 1, 0]


# all_nested_indices_w_nest_checks
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
def test_all_nested_indices_w_nest_checks(nest):
    indices = ivy.all_nested_indices(nest, True)
    assert indices[0] == ["a", 0, 0]
    assert indices[1] == ["a", 0]
    assert indices[2] == ["a", 1, 0]
    assert indices[3] == ["a", 1]
    assert indices[4] == ["a"]
    assert indices[5] == ["b", "c", 0, 0, 0]
    assert indices[6] == ["b", "c", 0, 0]
    assert indices[7] == ["b", "c", 0, 1, 0]
    assert indices[8] == ["b", "c", 0, 1]
    assert indices[9] == ["b", "c", 0]
    assert indices[10] == ["b", "c", 1, 0, 0]
    assert indices[11] == ["b", "c", 1, 0]
    assert indices[12] == ["b", "c", 1, 1, 0]
    assert indices[13] == ["b", "c", 1, 1]
    assert indices[14] == ["b", "c", 1]
    assert indices[15] == ["b", "c"]
    assert indices[16] == ["b"]


# all_nested_indices_w_extra_nest_types
def test_all_nested_indices_w_extra_nest_types():
    nest = {"a": ivy.array([[0], [1]]), "b": {"c": ivy.array([[[2], [4]], [[6], [8]]])}}
    indices = ivy.all_nested_indices(nest, extra_nest_types=ivy.Array)
    assert indices[0] == ["a", 0, 0]
    assert indices[1] == ["a", 1, 0]
    assert indices[2] == ["b", "c", 0, 0, 0]
    assert indices[3] == ["b", "c", 0, 1, 0]
    assert indices[4] == ["b", "c", 1, 0, 0]
    assert indices[5] == ["b", "c", 1, 1, 0]


# copy_nest
def test_copy_nest():
    nest = {
        "a": [ivy.array([0]), ivy.array([1])],
        "b": {"c": [ivy.array([[2], [4]]), ivy.array([[6], [8]])]},
    }
    nest_copy = ivy.copy_nest(nest)

    # copied nests
    assert nest["a"] is not nest_copy["a"]
    assert nest["b"] is not nest_copy["b"]
    assert nest["b"]["c"] is not nest_copy["b"]["c"]

    # non-copied arrays
    assert nest["a"][0] is nest_copy["a"][0]
    assert nest["a"][1] is nest_copy["a"][1]
    assert nest["b"]["c"][0] is nest_copy["b"]["c"][0]
    assert nest["b"]["c"][1] is nest_copy["b"]["c"][1]

    from collections import namedtuple

    NAMEDTUPLE = namedtuple("OutNamedTuple", ["x", "y"])
    nest = NAMEDTUPLE(x=ivy.array([1.0]), y=ivy.array([2.0]))
    copied_nest = ivy.copy_nest(nest, include_derived=True)
    assert isinstance(copied_nest, NAMEDTUPLE)


# copy_nest_w_extra_nest_types
def test_copy_nest_w_extra_nest_types():
    nest = {
        "a": [ivy.array([0]), ivy.array([1])],
        "b": {"c": [ivy.array([2, 4]), ivy.array([6, 8])]},
    }
    nest_copy = ivy.copy_nest(nest, extra_nest_types=ivy.Array)

    # copied nests
    assert nest["a"] is not nest_copy["a"]
    assert nest["b"] is not nest_copy["b"]
    assert nest["a"][0] is not nest_copy["a"][0]
    assert nest["a"][1] is not nest_copy["a"][1]
    assert nest["b"]["c"] is not nest_copy["b"]["c"]
    assert nest["b"]["c"][0] is not nest_copy["b"]["c"][0]
    assert nest["b"]["c"][1] is not nest_copy["b"]["c"][1]

    assert nest["a"][0][0] is not nest_copy["a"][0][0]
    assert nest["a"][1][0] is not nest_copy["a"][1][0]
    assert nest["b"]["c"][0][0] is not nest_copy["b"]["c"][0][0]
    assert nest["b"]["c"][0][0] is not nest_copy["b"]["c"][0][1]
    assert nest["b"]["c"][1][1] is not nest_copy["b"]["c"][1][1]
    assert nest["b"]["c"][1][1] is not nest_copy["b"]["c"][1][0]


# nested_multi_map
@pytest.mark.parametrize("func", [lambda x, _: x[0] - x[1]])
@pytest.mark.parametrize(
    "nests",
    [
        [
            np.asarray([-1.82, 1.25, -2.91, 0.109, 0.76, 1.7, 0.231, 4.45]),
            np.asarray([-3.98, -3.86, 7.94, 2.08, 9.3, 2.35, 9.37, 1.7]),
        ]
    ],
)
def test_nested_multi_map(func, nests):
    nests = ivy.nested_map(
        nests,
        lambda x: ivy.array(x) if isinstance(x, np.ndarray) else x,
        include_derived=True,
        shallow=False,
    )
    # without index_chains specification
    nested_multi_map_res = ivy.nested_multi_map(func, nests)

    # modify this to test for other functions
    nests_without_multi_map_res = nests[0] - nests[1]

    assert ivy.all_equal(nested_multi_map_res, nests_without_multi_map_res)


# prune_nest_at_index
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize(
    "index", [("a", 0, 0), ("a", 1, 0), ("b", "c", 0), ("b", "c", 1, 0)]
)
def test_prune_nest_at_index(nest, index):
    nest_copy = copy.deepcopy(nest)

    # handling cases where there is nothing to prune
    try:
        ivy.prune_nest_at_index(nest, index)
        _pnai(nest_copy, index)
    except Exception:
        warnings.warn("Nothing to delete.")

    assert nest == nest_copy


# prune_nest_at_indices
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize("indices", [(("a", 0, 0), ("b", "c", 0))])
def test_prune_nest_at_indices(nest, indices):
    nest_copy = copy.deepcopy(nest)

    def pnais(n, idxs):
        [_pnai(n, index) for index in idxs]

    # handling cases where there is nothing to prune
    try:
        ivy.prune_nest_at_indices(nest, indices)
        pnais(nest_copy, indices)
    except Exception:
        warnings.warn("Nothing to delete.")

    assert nest == nest_copy


# insert_into_nest_at_index
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize("index", [("a", 0, 0), ("a", 1, 0), ("b", "c", 0)])
@pytest.mark.parametrize("value", [1])
def test_insert_into_nest_index(nest, index, value):
    ivy.insert_into_nest_at_index(nest, index, value)

    assert ivy.index_nest(nest, index) == value


# insert_into_nest_at_indices
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize("indices", [(("a", 0, 0), ("b", "c", 1, 0))])
@pytest.mark.parametrize("values", [(1, 2)])
def test_insert_into_nest_at_indices(nest, indices, values):
    ivy.insert_into_nest_at_indices(nest, indices, values)

    def indices_nest(nest, indices):
        ret = tuple(ivy.index_nest(nest, index) for index in indices)

        return ret

    assert indices_nest(nest, indices) == values


# nested_map
@pytest.mark.parametrize("x", [{"a": [[0, 1], [2, 3]], "b": {"c": [[0], [1]]}}])
@pytest.mark.parametrize("fn", [lambda x: x**2])
@pytest.mark.parametrize("shallow", [True, False])
def test_nested_map(x, fn, shallow):
    x_copy = copy.deepcopy(x)
    result = ivy.nested_map(x, fn, shallow=shallow)
    map_nested_dicts(x_copy, fn)

    assert result == x_copy
    if shallow:
        assert x == x_copy
    else:
        assert x != x_copy


# nested_map_w_extra_nest_types
@pytest.mark.parametrize("fn", [lambda x: x**2])
def test_nested_map_w_extra_nest_types(fn):
    x = {"a": ivy.array([[0, 1], [2, 3]]), "b": {"c": ivy.array([[0], [1]])}}
    x_copy = copy.deepcopy(x)
    x = ivy.nested_map(x, fn, extra_nest_types=ivy.Array)
    map_nested_dicts(x_copy, fn)

    assert ivy.all(x_copy["a"] == x["a"])
    assert ivy.all(x_copy["b"]["c"] == x["b"]["c"])


# nested_any
@pytest.mark.parametrize("x", [{"a": [[0, 1], [2, 3]], "b": {"c": [[0], [1]]}}])
@pytest.mark.parametrize("fn", [lambda x: True if x % 2 == 0 else False])
def test_nested_any(x, fn):
    x_copy = copy.deepcopy(x)
    x_bool = ivy.nested_any(x, fn)
    map_nested_dicts(x_copy, fn)

    def is_true_any(ob):
        for k, v in ob.items():
            if isinstance(v, dict):
                is_true_any(v)
            if isinstance(v, list):
                for i, item in enumerate(v):
                    return item.count(True) == 1

    x_copy_bool = is_true_any(x_copy)

    assert x_copy_bool == x_bool


# nested_any_w_extra_nest_types
@pytest.mark.parametrize("fn", [lambda x: x % 2 == 0])
def test_nested_any_w_extra_nest_types(fn):
    x = {"a": ivy.array([[0, 1], [2, 3]]), "b": {"c": ivy.array([[0], [1]])}}
    x_copy = copy.deepcopy(x)
    x_bool = ivy.nested_any(x, fn, extra_nest_types=ivy.Array)

    def is_true_any(ob):
        for k, v in ob.items():
            if isinstance(v, dict):
                is_true_any(v)
            if isinstance(v, ivy.Array):
                return ivy.any(fn(v))

    x_copy_bool = is_true_any(x_copy)

    assert x_copy_bool == x_bool


# duplicate_array_index_chains
@pytest.mark.parametrize("x", [[-1.0]])
@pytest.mark.parametrize("y", [[1.0]])
@pytest.mark.parametrize(
    "nest", [[{"a": None, "b": {"c": None, "d": None}}, [None, None]]]
)
def test_duplicate_array_index_chains(nest, x, y):
    x = ivy.array(x)
    y = ivy.array(y)
    nest[0]["a"] = nest[0]["b"]["d"] = nest[1][0] = x
    nest[0]["b"]["c"] = nest[1][1] = y
    duplicate_index_chains = ivy.duplicate_array_index_chains(nest)
    assert duplicate_index_chains[0] == [[0, "a"], [0, "b", "d"], [1, 0]]
    assert duplicate_index_chains[1] == [[0, "b", "c"], [1, 1]]


# prune_empty
@pytest.mark.parametrize("nest", [{"a": [{}, {}], "b": {"c": [1], "d": []}}])
def test_prune_empty(nest):
    ret = ivy.prune_empty(ivy.copy_nest(nest))
    assert ret == {"b": {"c": [1]}}
