"""Collection of tests for unified general functions."""

# global
import copy
import warnings
import pytest

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
def test_index_nest(nest, index, device, call):
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
@pytest.mark.parametrize("value", [1])
def test_set_nest_at_index(nest, index, value, device, call):
    nest_copy = copy.deepcopy(nest)
    ivy.set_nest_at_index(nest, index, value)
    _snai(nest_copy, index, value)
    assert nest == nest_copy


# map_nest_at_index
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize(
    "index", [("a", 0, 0), ("a", 1, 0), ("b", "c", 0, 0, 0), ("b", "c", 1, 0, 0)]
)
@pytest.mark.parametrize("fn", [lambda x: x + 2, lambda x: x**2])
def test_map_nest_at_index(nest, index, fn, device, call):
    nest_copy = copy.deepcopy(nest)
    ivy.map_nest_at_index(nest, index, fn)
    _mnai(nest_copy, index, fn)
    assert nest == nest_copy


# multi_index_nest
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": (((2,), (4,)), ((6,), (8,)))}}]
)
@pytest.mark.parametrize(
    "multi_indices", [(("a", 0, 0), ("a", 1, 0)), (("b", "c", 0), ("b", "c", 1, 0))]
)
def test_multi_index_nest(nest, multi_indices, device, call):
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
def test_set_nest_at_indices(nest, indices, values, device, call):
    nest_copy = copy.deepcopy(nest)
    ivy.set_nest_at_indices(nest, indices, values)

    def snais(n, idxs, vs):
        [_snai(n, index, value) for index, value in zip(idxs, vs)]

    snais(nest_copy, indices, values)

    assert nest == nest_copy


# map_nest_at_indices
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize(
    "indices", [(("a", 0, 0), ("a", 1, 0)), (("b", "c", 0, 0, 0), ("b", "c", 1, 0, 0))]
)
@pytest.mark.parametrize("fn", [lambda x: x + 2, lambda x: x**2])
def test_map_nest_at_indices(nest, indices, fn, device, call):
    nest_copy = copy.deepcopy(nest)
    ivy.map_nest_at_indices(nest, indices, fn)

    def mnais(n, idxs, vs):
        [_mnai(n, index, fn) for index in idxs]

    mnais(nest_copy, indices, fn)

    assert nest == nest_copy


# nested_indices_where
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
def test_nested_indices_where(nest, device, call):
    indices = ivy.nested_indices_where(nest, lambda x: x < 5)
    assert indices[0] == ["a", 0, 0]
    assert indices[1] == ["a", 1, 0]
    assert indices[2] == ["b", "c", 0, 0, 0]
    assert indices[3] == ["b", "c", 0, 1, 0]


# nested_indices_where_w_nest_checks
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
def test_nested_indices_where_w_nest_checks(nest, device, call):
    indices = ivy.nested_indices_where(
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


# all_nested_indices
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
def test_all_nested_indices(nest, device, call):
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
def test_all_nested_indices_w_nest_checks(nest, device, call):
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


# copy_nest
def test_copy_nest(device, call):

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


# nested_multi_map
@pytest.mark.parametrize("func", [lambda x, _: x[0] - x[1]])
@pytest.mark.parametrize(
    "nests",
    [
        [
            ivy.array([-1.82, 1.25, -2.91, 0.109, 0.76, 1.7, 0.231, 4.45]),
            ivy.array([-3.98, -3.86, 7.94, 2.08, 9.3, 2.35, 9.37, 1.7]),
        ]
    ],
)
def test_nested_multi_map(func, nests, device, call, fw):
    # without key_chains specification
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
def test_prune_nest_at_index(nest, index, device, call):
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
def test_prune_nest_at_indices(nest, indices, device, call):
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
def test_insert_into_nest_index(nest, index, value, device, call):

    ivy.insert_into_nest_at_index(nest, index, value)

    assert ivy.index_nest(nest, index) == value


# insert_into_nest_at_indices
@pytest.mark.parametrize(
    "nest", [{"a": [[0], [1]], "b": {"c": [[[2], [4]], [[6], [8]]]}}]
)
@pytest.mark.parametrize("indices", [(("a", 0, 0), ("b", "c", 1, 0))])
@pytest.mark.parametrize("values", [(1, 2)])
def test_insert_into_nest_at_indices(nest, indices, values, device, call):

    ivy.insert_into_nest_at_indices(nest, indices, values)

    def indices_nest(nest, indices):
        ret = tuple(ivy.index_nest(nest, index) for index in indices)

        return ret

    assert indices_nest(nest, indices) == values


# nested_map
@pytest.mark.parametrize("x", [{"a": [[0, 1], [2, 3]], "b": {"c": [[0], [1]]}}])
@pytest.mark.parametrize("fn", [lambda x: x**2])
def test_nested_map(x, fn):
    x_copy = copy.deepcopy(x)
    x = ivy.nested_map(x, fn)
    map_nested_dicts(x_copy, fn)

    assert x_copy == x


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
