# global
import os
import pytest
import random
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers
from ivy.core.container import Container


def test_container_from_dict(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    assert container['a'] == ivy.array([1])
    assert container.a == ivy.array([1])
    assert container['b']['c'] == ivy.array([2])
    assert container.b.c == ivy.array([2])
    assert container['b']['d'] == ivy.array([3])
    assert container.b.d == ivy.array([3])


def test_container_expand_dims(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)

    # without key_chains specification
    container_expanded_dims = container.expand_dims(0)
    assert (container_expanded_dims['a'] == ivy.array([[1]]))[0, 0]
    assert (container_expanded_dims.a == ivy.array([[1]]))[0, 0]
    assert (container_expanded_dims['b']['c'] == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims.b.c == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims['b']['d'] == ivy.array([[3]]))[0, 0]
    assert (container_expanded_dims.b.d == ivy.array([[3]]))[0, 0]

    # with key_chains to apply
    container_expanded_dims = container.expand_dims(0, ['a', 'b/c'])
    assert (container_expanded_dims['a'] == ivy.array([[1]]))[0, 0]
    assert (container_expanded_dims.a == ivy.array([[1]]))[0, 0]
    assert (container_expanded_dims['b']['c'] == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims.b.c == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims['b']['d'] == ivy.array([3]))[0]
    assert (container_expanded_dims.b.d == ivy.array([3]))[0]

    # with key_chains to apply pruned
    container_expanded_dims = container.expand_dims(0, ['a', 'b/c'], prune_unapplied=True)
    assert (container_expanded_dims['a'] == ivy.array([[1]]))[0, 0]
    assert (container_expanded_dims.a == ivy.array([[1]]))[0, 0]
    assert (container_expanded_dims['b']['c'] == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims.b.c == ivy.array([[2]]))[0, 0]
    assert 'b/d' not in container_expanded_dims

    # with key_chains to not apply
    container_expanded_dims = container.expand_dims(0, Container({'a': None, 'b': {'d': None}}), to_apply=False)
    assert (container_expanded_dims['a'] == ivy.array([1]))[0]
    assert (container_expanded_dims.a == ivy.array([1]))[0]
    assert (container_expanded_dims['b']['c'] == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims.b.c == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims['b']['d'] == ivy.array([3]))[0]
    assert (container_expanded_dims.b.d == ivy.array([3]))[0]

    # with key_chains to not apply pruned
    container_expanded_dims = container.expand_dims(0, Container({'a': None, 'b': {'d': None}}), to_apply=False,
                                                    prune_unapplied=True)
    assert 'a' not in container_expanded_dims
    assert (container_expanded_dims['b']['c'] == ivy.array([[2]]))[0, 0]
    assert (container_expanded_dims.b.c == ivy.array([[2]]))[0, 0]
    assert 'b/d' not in container_expanded_dims


def test_container_gather(dev_str, call):
    dict_in = {'a': ivy.array([1, 2, 3, 4, 5, 6]),
               'b': {'c': ivy.array([2, 3, 4, 5]), 'd': ivy.array([10, 9, 8, 7, 6])}}
    container = Container(dict_in)

    # without key_chains specification
    container_gathered = container.gather(ivy.array([1, 3]))
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([9, 7]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([9, 7]))

    # with key_chains to apply
    container_gathered = container.gather(ivy.array([1, 3]), -1, ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([10, 9, 8, 7, 6]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([10, 9, 8, 7, 6]))

    # with key_chains to apply pruned
    container_gathered = container.gather(ivy.array([1, 3]), -1, ['a', 'b/c'], prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert 'b/d' not in container_gathered

    # with key_chains to not apply
    container_gathered = container.gather(ivy.array([1, 3]), -1, Container({'a': None, 'b': {'d': None}}),
                                          to_apply=False)
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([1, 2, 3, 4, 5, 6]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([1, 2, 3, 4, 5, 6]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([10, 9, 8, 7, 6]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([10, 9, 8, 7, 6]))

    # with key_chains to not apply pruned
    container_gathered = container.gather(ivy.array([1, 3]), -1, Container({'a': None, 'b': {'d': None}}),
                                          to_apply=False, prune_unapplied=True)
    assert 'a' not in container_gathered
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert 'b/d' not in container_gathered


def test_container_gather_nd(dev_str, call):
    dict_in = {'a': ivy.array([[[1, 2], [3, 4]],
                               [[5, 6], [7, 8]]]),
               'b': {'c': ivy.array([[[8, 7], [6, 5]],
                                     [[4, 3], [2, 1]]]), 'd': ivy.array([[[2, 4], [6, 8]],
                                                                         [[10, 12], [14, 16]]])}}
    container = Container(dict_in)

    # without key_chains specification
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]]))
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([[6, 8], [10, 12]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([[6, 8], [10, 12]]))

    # with key_chains to apply
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]]), ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([[[2, 4], [6, 8]],
                                                                             [[10, 12], [14, 16]]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([[[2, 4], [6, 8]],
                                                                       [[10, 12], [14, 16]]]))

    # with key_chains to apply pruned
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]]), ['a', 'b/c'], prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert 'b/d' not in container_gathered

    # with key_chains to not apply
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]]), Container({'a': None, 'b': {'d': None}}),
                                             to_apply=False)
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([[[1, 2], [3, 4]],
                                                                        [[5, 6], [7, 8]]]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([[[1, 2], [3, 4]],
                                                                     [[5, 6], [7, 8]]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([[[2, 4], [6, 8]],
                                                                             [[10, 12], [14, 16]]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([[[2, 4], [6, 8]],
                                                                       [[10, 12], [14, 16]]]))

    # with key_chains to not apply pruned
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]]), Container({'a': None, 'b': {'d': None}}),
                                             to_apply=False, prune_unapplied=True)
    assert 'a' not in container_gathered
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert 'b/d' not in container_gathered


def test_container_repeat(dev_str, call):
    if call is helpers.mx_call:
        # MXNet does not support repeats specified as array
        pytest.skip()
    dict_in = {'a': ivy.array([[0., 1., 2., 3.]]),
               'b': {'c': ivy.array([[5., 10., 15., 20.]]), 'd': ivy.array([[10., 9., 8., 7.]])}}
    container = Container(dict_in)

    # without key_chains specification
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3]), -1)
    assert np.allclose(ivy.to_numpy(container_repeated['a']), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['d']), np.array([[10., 10., 9., 7., 7., 7.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.d), np.array([[10., 10., 9., 7., 7., 7.]]))

    # with key_chains to apply
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3]), -1, ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_repeated['a']), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['d']), np.array([[10., 9., 8., 7.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.d), np.array([[10., 9., 8., 7.]]))

    # with key_chains to apply pruned
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3]), -1, ['a', 'b/c'], prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_repeated['a']), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert 'b/d' not in container_repeated

    # with key_chains to not apply
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3]), -1, Container({'a': None, 'b': {'d': None}}),
                                          to_apply=False)
    assert np.allclose(ivy.to_numpy(container_repeated['a']), np.array([[0., 1., 2., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a), np.array([[0., 1., 2., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['d']), np.array([[10., 9., 8., 7.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.d), np.array([[10., 9., 8., 7.]]))

    # with key_chains to not apply pruned
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3]), -1, Container({'a': None, 'b': {'d': None}}),
                                          to_apply=False, prune_unapplied=True)
    assert 'a' not in container_repeated
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert 'b/d' not in container_repeated


def test_container_stop_gradients(dev_str, call):
    dict_in = {'a': ivy.variable(ivy.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])),
               'b': {'c': ivy.variable(ivy.array([[[8., 7.], [6., 5.]], [[4., 3.], [2., 1.]]])),
                     'd': ivy.variable(ivy.array([[[2., 4.], [6., 8.]], [[10., 12.], [14., 16.]]]))}}
    container = Container(dict_in)
    if call is not helpers.np_call:
        # Numpy does not support variables or gradients
        assert ivy.is_variable(container['a'])
        assert ivy.is_variable(container.a)
        assert ivy.is_variable(container['b']['c'])
        assert ivy.is_variable(container.b.c)
        assert ivy.is_variable(container['b']['d'])
        assert ivy.is_variable(container.b.d)
    container_gathered = container.stop_gradients()
    assert ivy.is_array(container['a'])
    assert ivy.is_array(container.a)
    assert ivy.is_array(container['b']['c'])
    assert ivy.is_array(container.b.c)
    assert ivy.is_array(container['b']['d'])
    assert ivy.is_array(container.b.d)


def test_container_has_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    assert container.has_key_chain('a')
    assert container.has_key_chain('b')
    assert container.has_key_chain('b/c')
    assert container.has_key_chain('b/d')
    assert not container.has_key_chain('b/e')
    assert not container.has_key_chain('c')


def test_container_at_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)

    # explicit function call
    sub_container = container.at_key_chain('b')
    assert (sub_container['c'] == ivy.array([2]))[0]
    sub_container = container.at_key_chain('b/c')
    assert (sub_container == ivy.array([2]))[0]

    # overridden built-in function call
    sub_container = container['b']
    assert (sub_container['c'] == ivy.array([2]))[0]
    sub_container = container['b/c']
    assert (sub_container == ivy.array([2]))[0]


def test_container_at_key_chains(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    target_cont = Container({'a': True, 'b': {'c': True}})
    new_container = container.at_key_chains(target_cont)
    assert (new_container['a'] == ivy.array([1]))[0]
    assert (new_container['b']['c'] == ivy.array([2]))[0]
    assert 'd' not in new_container['b']
    new_container = container.at_key_chains(['b/c', 'b/d'])
    assert 'a' not in new_container
    assert (new_container['b']['c'] == ivy.array([2]))[0]
    assert (new_container['b']['d'] == ivy.array([3]))[0]
    new_container = container.at_key_chains('b/c')
    assert 'a' not in new_container
    assert (new_container['b']['c'] == ivy.array([2]))[0]
    assert 'd' not in new_container['b']


# noinspection PyUnresolvedReferences
def test_container_set_at_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container_orig = Container(dict_in)

    # explicit function call
    container = container_orig.copy()
    container.set_at_key_chain('b/e', ivy.array([4]))
    assert (container['a'] == ivy.array([1]))[0]
    assert (container['b']['c'] == ivy.array([2]))[0]
    assert (container['b']['d'] == ivy.array([3]))[0]
    assert (container['b']['e'] == ivy.array([4]))[0]
    container.set_at_key_chain('f', ivy.array([5]))
    assert (container['a'] == ivy.array([1]))[0]
    assert (container['b']['c'] == ivy.array([2]))[0]
    assert (container['b']['d'] == ivy.array([3]))[0]
    assert (container['b']['e'] == ivy.array([4]))[0]
    assert (container['f'] == ivy.array([5]))[0]

    # overridden built-in function call
    container = container_orig.copy()
    assert 'b/e' not in container
    container['b/e'] = ivy.array([4])
    assert (container['a'] == ivy.array([1]))[0]
    assert (container['b']['c'] == ivy.array([2]))[0]
    assert (container['b']['d'] == ivy.array([3]))[0]
    assert (container['b']['e'] == ivy.array([4]))[0]
    assert 'f' not in container
    container['f'] = ivy.array([5])
    assert (container['a'] == ivy.array([1]))[0]
    assert (container['b']['c'] == ivy.array([2]))[0]
    assert (container['b']['d'] == ivy.array([3]))[0]
    assert (container['b']['e'] == ivy.array([4]))[0]
    assert (container['f'] == ivy.array([5]))[0]


def test_container_set_at_key_chains(dev_str, call):
    container = Container({'a': ivy.array([1]),
                           'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    target_container = Container({'a': ivy.array([4]),
                                  'b': {'d': ivy.array([5])}})
    new_container = container.set_at_key_chains(target_container)
    assert (new_container['a'] == ivy.array([4]))[0]
    assert (new_container['b']['c'] == ivy.array([2]))[0]
    assert (new_container['b']['d'] == ivy.array([5]))[0]
    target_container = Container({'b': {'c': ivy.array([7])}})
    new_container = container.set_at_key_chains(target_container)
    assert (new_container['a'] == ivy.array([1]))[0]
    assert (new_container['b']['c'] == ivy.array([7]))[0]
    assert (new_container['b']['d'] == ivy.array([3]))[0]


def test_container_prune_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': None}}
    container = Container(dict_in)
    container_pruned = container.prune_key_chain('b/c')
    assert (container_pruned['a'] == ivy.array([[1]]))[0, 0]
    assert (container_pruned.a == ivy.array([[1]]))[0, 0]
    assert (container_pruned['b']['d'] is None)
    assert (container_pruned.b.d is None)
    assert ('c' not in container_pruned['b'].keys())

    def _test_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)

    container_pruned = container.prune_key_chain('b')
    assert (container_pruned['a'] == ivy.array([[1]]))[0, 0]
    assert (container_pruned.a == ivy.array([[1]]))[0, 0]
    assert ('b' not in container_pruned.keys())

    def _test_exception(container_in):
        try:
            _ = container_in.b
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)


def test_container_prune_key_chains(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_pruned = container.prune_key_chains(['a', 'b/c'])
    assert 'a' not in container_pruned
    assert (container_pruned['b']['d'] == ivy.array([[3]]))[0, 0]
    assert (container_pruned.b.d == ivy.array([[3]]))[0, 0]
    assert 'c' not in container_pruned['b']

    def _test_a_exception(container_in):
        try:
            _ = container_in.a
            return False
        except AttributeError:
            return True

    def _test_bc_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_a_exception(container_pruned)
    assert _test_bc_exception(container_pruned)

    container_pruned = container.prune_key_chains(Container({'a': True, 'b': {'c': True}}))
    assert 'a' not in container_pruned
    assert (container_pruned['b']['d'] == ivy.array([[3]]))[0, 0]
    assert (container_pruned.b.d == ivy.array([[3]]))[0, 0]
    assert 'c' not in container_pruned['b']
    assert _test_a_exception(container_pruned)
    assert _test_bc_exception(container_pruned)


def test_container_prune_empty(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': {}, 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_pruned = container.prune_empty()
    assert (container_pruned['a'] == ivy.array([[1]]))[0, 0]
    assert (container_pruned.a == ivy.array([[1]]))[0, 0]
    assert (container_pruned['b']['d'] == ivy.array([[3]]))[0, 0]
    assert (container_pruned.b.d == ivy.array([[3]]))[0, 0]
    assert ('c' not in container_pruned['b'].keys())

    def _test_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)


def test_container_contains(dev_str, call):
    dict_in = {'a': ivy.array([0.]),
               'b': {'c': ivy.array([1.]), 'd': ivy.array([2.])}}
    container = Container(dict_in)
    assert 'a' in container
    assert 'b' in container
    assert 'c' not in container
    assert 'b/c' in container
    assert 'd' not in container
    assert 'b/d' in container


def test_container_shuffle(dev_str, call):
    if call is helpers.tf_graph_call:
        # tf.random.set_seed is not compiled. The shuffle is then not aligned between container items.
        pytest.skip()
    dict_in = {'a': ivy.array([1, 2, 3]),
               'b': {'c': ivy.array([1, 2, 3]), 'd': ivy.array([1, 2, 3])}}
    container = Container(dict_in)

    # without key_chains specification
    container_shuffled = container.shuffle(0)
    data = ivy.array([1, 2, 3])
    ivy.core.random.seed()
    shuffled_data = ivy.core.random.shuffle(data)
    assert np.array(container_shuffled['a'] == shuffled_data).all()
    assert np.array(container_shuffled.a == shuffled_data).all()
    assert np.array(container_shuffled['b']['c'] == shuffled_data).all()
    assert np.array(container_shuffled.b.c == shuffled_data).all()
    assert np.array(container_shuffled['b']['d'] == shuffled_data).all()
    assert np.array(container_shuffled.b.d == shuffled_data).all()

    # with key_chains to apply
    container_shuffled = container.shuffle(0, ['a', 'b/c'])
    data = ivy.array([1, 2, 3])
    ivy.core.random.seed()
    shuffled_data = ivy.core.random.shuffle(data)
    assert np.array(container_shuffled['a'] == shuffled_data).all()
    assert np.array(container_shuffled.a == shuffled_data).all()
    assert np.array(container_shuffled['b']['c'] == shuffled_data).all()
    assert np.array(container_shuffled.b.c == shuffled_data).all()
    assert np.array(container_shuffled['b']['d'] == data).all()
    assert np.array(container_shuffled.b.d == data).all()

    # with key_chains to apply pruned
    container_shuffled = container.shuffle(0, ['a', 'b/c'], prune_unapplied=True)
    data = ivy.array([1, 2, 3])
    ivy.core.random.seed()
    shuffled_data = ivy.core.random.shuffle(data)
    assert np.array(container_shuffled['a'] == shuffled_data).all()
    assert np.array(container_shuffled.a == shuffled_data).all()
    assert np.array(container_shuffled['b']['c'] == shuffled_data).all()
    assert np.array(container_shuffled.b.c == shuffled_data).all()
    assert 'b/d' not in container_shuffled

    # with key_chains to not apply pruned
    container_shuffled = container.shuffle(0, Container({'a': None, 'b': {'d': None}}), to_apply=False)
    data = ivy.array([1, 2, 3])
    ivy.core.random.seed()
    shuffled_data = ivy.core.random.shuffle(data)
    assert np.array(container_shuffled['a'] == data).all()
    assert np.array(container_shuffled.a == data).all()
    assert np.array(container_shuffled['b']['c'] == shuffled_data).all()
    assert np.array(container_shuffled.b.c == shuffled_data).all()
    assert np.array(container_shuffled['b']['d'] == data).all()
    assert np.array(container_shuffled.b.d == data).all()

    # with key_chains to not apply pruned
    container_shuffled = container.shuffle(0, Container({'a': None, 'b': {'d': None}}), to_apply=False,
                                           prune_unapplied=True)
    data = ivy.array([1, 2, 3])
    ivy.core.random.seed()
    shuffled_data = ivy.core.random.shuffle(data)
    assert 'a' not in container_shuffled
    assert np.array(container_shuffled['b']['c'] == shuffled_data).all()
    assert np.array(container_shuffled.b.c == shuffled_data).all()
    assert 'b/d' not in container_shuffled


def test_container_to_iterator(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_iterator = container.to_iterator()
    for (key, value), expected_value in zip(container_iterator,
                                            [ivy.array([1]), ivy.array([2]), ivy.array([3])]):
        assert value == expected_value


def test_container_to_flat_list(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_flat_list = container.to_flat_list()
    for value, expected_value in zip(container_flat_list, [ivy.array([1]), ivy.array([2]), ivy.array([3])]):
        assert value == expected_value


def test_container_from_flat_list(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    flat_list = [4, 5, 6]
    container = container.from_flat_list(flat_list)
    assert container['a'] == ivy.array([4])
    assert container.a == ivy.array([4])
    assert container['b']['c'] == ivy.array([5])
    assert container.b.c == ivy.array([5])
    assert container['b']['d'] == ivy.array([6])
    assert container.b.d == ivy.array([6])


def test_container_map(dev_str, call):

    # without key_chains specification
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    container_iterator = container.map(lambda x, _: x + 1).to_iterator()
    for (key, value), expected_value in zip(container_iterator,
                                            [ivy.array([2]), ivy.array([3]), ivy.array([4])]):
        assert call(lambda x: x, value) == call(lambda x: x, expected_value)

    # with key_chains to apply
    container_mapped = container.map(lambda x, _: x + 1, ['a', 'b/c'])
    assert (container_mapped['a'] == ivy.array([[2]]))[0]
    assert (container_mapped.a == ivy.array([[2]]))[0]
    assert (container_mapped['b']['c'] == ivy.array([[3]]))[0]
    assert (container_mapped.b.c == ivy.array([[3]]))[0]
    assert (container_mapped['b']['d'] == ivy.array([[3]]))[0]
    assert (container_mapped.b.d == ivy.array([[3]]))[0]

    # with key_chains to apply pruned
    container_mapped = container.map(lambda x, _: x + 1, ['a', 'b/c'], prune_unapplied=True)
    assert (container_mapped['a'] == ivy.array([[2]]))[0]
    assert (container_mapped.a == ivy.array([[2]]))[0]
    assert (container_mapped['b']['c'] == ivy.array([[3]]))[0]
    assert (container_mapped.b.c == ivy.array([[3]]))[0]
    assert 'b/d' not in container_mapped

    # with key_chains to not apply
    container_mapped = container.map(lambda x, _: x + 1, Container({'a': None, 'b': {'d': None}}), to_apply=False)
    assert (container_mapped['a'] == ivy.array([[1]]))[0]
    assert (container_mapped.a == ivy.array([[1]]))[0]
    assert (container_mapped['b']['c'] == ivy.array([[3]]))[0]
    assert (container_mapped.b.c == ivy.array([[3]]))[0]
    assert (container_mapped['b']['d'] == ivy.array([[3]]))[0]
    assert (container_mapped.b.d == ivy.array([[3]]))[0]

    # with key_chains to not apply pruned
    container_mapped = container.map(lambda x, _: x + 1, Container({'a': None, 'b': {'d': None}}), to_apply=False,
                                     prune_unapplied=True)
    assert 'a' not in container_mapped
    assert (container_mapped['b']['c'] == ivy.array([[3]]))[0]
    assert (container_mapped.b.c == ivy.array([[3]]))[0]
    assert 'b/d' not in container_mapped


def test_container_to_random(dev_str, call):
    dict_in = {'a': ivy.array([1.]),
               'b': {'c': ivy.array([2.]), 'd': ivy.array([3.])}}
    container = Container(dict_in)
    random_container = container.to_random()
    for (key, value), orig_value in zip(random_container.to_iterator(),
                                        [ivy.array([2]), ivy.array([3]), ivy.array([4])]):
        assert call(ivy.shape, value) == call(ivy.shape, orig_value)


def test_container_dtype(dev_str, call):
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2.]), 'd': ivy.array([3])}}
    container = Container(dict_in)
    dtype_container = container.dtype()
    for (key, value), expected_value in zip(dtype_container.to_iterator(),
                                            [ivy.array([1]).dtype,
                                             ivy.array([2.]).dtype,
                                             ivy.array([3]).dtype]):
        assert value == expected_value


def test_container_with_entries_as_lists(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # to_list() requires eager execution
        pytest.skip()
    dict_in = {'a': ivy.array([1]),
               'b': {'c': ivy.array([2.]), 'd': 'some string'}}
    container = Container(dict_in)
    container_w_list_entries = container.with_entries_as_lists()
    for (key, value), expected_value in zip(container_w_list_entries.to_iterator(),
                                            [[1],
                                             [2.],
                                             'some string']):
        assert value == expected_value


def test_container_reshape(dev_str, call):
    container = Container({'a': ivy.array([[1.]]),
                           'b': {'c': ivy.array([[3.], [4.]]), 'd': ivy.array([[5.], [6.], [7.]])}})
    new_shapes = Container({'a': (1,),
                           'b': {'c': (1, 2, 1), 'd': (3, 1, 1)}})
    container_reshaped = container.reshape(new_shapes)
    assert list(container_reshaped['a'].shape) == [1]
    assert list(container_reshaped.a.shape) == [1]
    assert list(container_reshaped['b']['c'].shape) == [1, 2, 1]
    assert list(container_reshaped.b.c.shape) == [1, 2, 1]
    assert list(container_reshaped['b']['d'].shape) == [3, 1, 1]
    assert list(container_reshaped.b.d.shape) == [3, 1, 1]


def test_container_slice(dev_str, call):
    dict_in = {'a': ivy.array([[0.], [1.]]),
               'b': {'c': ivy.array([[1.], [2.]]), 'd': ivy.array([[2.], [3.]])}}
    container = Container(dict_in)
    container0 = container[0]
    container1 = container[1]
    assert np.array_equal(container0['a'], ivy.array([0.]))
    assert np.array_equal(container0.a, ivy.array([0.]))
    assert np.array_equal(container0['b']['c'], ivy.array([1.]))
    assert np.array_equal(container0.b.c, ivy.array([1.]))
    assert np.array_equal(container0['b']['d'], ivy.array([2.]))
    assert np.array_equal(container0.b.d, ivy.array([2.]))
    assert np.array_equal(container1['a'], ivy.array([1.]))
    assert np.array_equal(container1.a, ivy.array([1.]))
    assert np.array_equal(container1['b']['c'], ivy.array([2.]))
    assert np.array_equal(container1.b.c, ivy.array([2.]))
    assert np.array_equal(container1['b']['d'], ivy.array([3.]))
    assert np.array_equal(container1.b.d, ivy.array([3.]))


def test_container_slice_via_key(dev_str, call):
    dict_in = {'a': {'x': ivy.array([0.]),
                     'y': ivy.array([1.])},
               'b': {'c': {'x': ivy.array([1.]),
                           'y': ivy.array([2.])},
                     'd': {'x': ivy.array([2.]),
                           'y': ivy.array([3.])}}}
    container = Container(dict_in)
    containerx = container.slice_via_key('x')
    containery = container.slice_via_key('y')
    assert np.array_equal(containerx['a'], ivy.array([0.]))
    assert np.array_equal(containerx.a, ivy.array([0.]))
    assert np.array_equal(containerx['b']['c'], ivy.array([1.]))
    assert np.array_equal(containerx.b.c, ivy.array([1.]))
    assert np.array_equal(containerx['b']['d'], ivy.array([2.]))
    assert np.array_equal(containerx.b.d, ivy.array([2.]))
    assert np.array_equal(containery['a'], ivy.array([1.]))
    assert np.array_equal(containery.a, ivy.array([1.]))
    assert np.array_equal(containery['b']['c'], ivy.array([2.]))
    assert np.array_equal(containery.b.c, ivy.array([2.]))
    assert np.array_equal(containery['b']['d'], ivy.array([3.]))
    assert np.array_equal(containery.b.d, ivy.array([3.]))


def test_container_to_and_from_disk_as_hdf5(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.hdf5'
    dict_in_1 = {'a': ivy.array([np.float32(1.)]),
                 'b': {'c': ivy.array([np.float32(2.)]), 'd': ivy.array([np.float32(3.)])}}
    container1 = Container(dict_in_1)
    dict_in_2 = {'a': ivy.array([np.float32(1.), np.float32(1.)]),
                 'b': {'c': ivy.array([np.float32(2.), np.float32(2.)]),
                       'd': ivy.array([np.float32(3.), np.float32(3.)])}}
    container2 = Container(dict_in_2)

    # saving
    container1.to_disk_as_hdf5(save_filepath, max_batch_size=2)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_hdf5(save_filepath, slice(1))
    assert np.array_equal(loaded_container.a, container1.a)
    assert np.array_equal(loaded_container.b.c, container1.b.c)
    assert np.array_equal(loaded_container.b.d, container1.b.d)

    # appending
    container1.to_disk_as_hdf5(save_filepath, max_batch_size=2, starting_index=1)
    assert os.path.exists(save_filepath)

    # loading after append
    loaded_container = Container.from_disk_as_hdf5(save_filepath)
    assert np.array_equal(loaded_container.a, container2.a)
    assert np.array_equal(loaded_container.b.c, container2.b.c)
    assert np.array_equal(loaded_container.b.d, container2.b.d)

    # load slice
    loaded_sliced_container = Container.from_disk_as_hdf5(save_filepath, slice(1, 2))
    assert np.array_equal(loaded_sliced_container.a, container1.a)
    assert np.array_equal(loaded_sliced_container.b.c, container1.b.c)
    assert np.array_equal(loaded_sliced_container.b.d, container1.b.d)

    # file size
    file_size, batch_size = Container.h5_file_size(save_filepath)
    assert file_size == 6 * np.dtype(np.float32).itemsize
    assert batch_size == 2

    os.remove(save_filepath)


def test_container_to_disk_shuffle_and_from_disk_as_hdf5(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.hdf5'
    dict_in = {'a': ivy.array([1, 2, 3]),
               'b': {'c': ivy.array([1, 2, 3]), 'd': ivy.array([1, 2, 3])}}
    container = Container(dict_in)

    # saving
    container.to_disk_as_hdf5(save_filepath, max_batch_size=3)
    assert os.path.exists(save_filepath)

    # shuffling
    Container.shuffle_h5_file(save_filepath)

    # loading
    container_shuffled = Container.from_disk_as_hdf5(save_filepath, slice(3))

    # testing
    data = np.array([1, 2, 3])
    random.seed(0)
    random.shuffle(data)

    assert (ivy.to_numpy(container_shuffled['a']) == data).all()
    assert (ivy.to_numpy(container_shuffled.a) == data).all()
    assert (ivy.to_numpy(container_shuffled['b']['c']) == data).all()
    assert (ivy.to_numpy(container_shuffled.b.c) == data).all()
    assert (ivy.to_numpy(container_shuffled['b']['d']) == data).all()
    assert (ivy.to_numpy(container_shuffled.b.d) == data).all()

    os.remove(save_filepath)


def test_container_to_and_from_disk_as_pickled(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.pickled'
    dict_in = {'a': ivy.array([np.float32(1.)]),
               'b': {'c': ivy.array([np.float32(2.)]), 'd': ivy.array([np.float32(3.)])}}
    container = Container(dict_in)

    # saving
    container.to_disk_as_pickled(save_filepath)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_pickled(save_filepath)
    assert np.array_equal(loaded_container.a, container.a)
    assert np.array_equal(loaded_container.b.c, container.b.c)
    assert np.array_equal(loaded_container.b.d, container.b.d)

    os.remove(save_filepath)


def test_container_to_and_from_disk_as_json(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.json'
    dict_in = {'a': 1.274e-7, 'b': {'c': True, 'd': ivy.array([np.float32(3.)])}}
    container = Container(dict_in)

    # saving
    container.to_disk_as_json(save_filepath)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_json(save_filepath)
    assert np.array_equal(loaded_container.a, container.a)
    assert np.array_equal(loaded_container.b.c, container.b.c)
    assert isinstance(loaded_container.b.d, str)

    os.remove(save_filepath)


def test_container_positive(dev_str, call):
    container = +Container({'a': ivy.array([1]), 'b': {'c': ivy.array([-2]), 'd': ivy.array([3])}})
    assert container['a'] == ivy.array([1])
    assert container.a == ivy.array([1])
    assert container['b']['c'] == ivy.array([-2])
    assert container.b.c == ivy.array([-2])
    assert container['b']['d'] == ivy.array([3])
    assert container.b.d == ivy.array([3])


def test_container_negative(dev_str, call):
    container = -Container({'a': ivy.array([1]), 'b': {'c': ivy.array([-2]), 'd': ivy.array([3])}})
    assert container['a'] == ivy.array([-1])
    assert container.a == ivy.array([-1])
    assert container['b']['c'] == ivy.array([2])
    assert container.b.c == ivy.array([2])
    assert container['b']['d'] == ivy.array([-3])
    assert container.b.d == ivy.array([-3])


def test_container_pow(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([4]), 'd': ivy.array([6])}})
    container = container_a ** container_b
    assert container['a'] == ivy.array([1])
    assert container.a == ivy.array([1])
    assert container['b']['c'] == ivy.array([16])
    assert container.b.c == ivy.array([16])
    assert container['b']['d'] == ivy.array([729])
    assert container.b.d == ivy.array([729])


def test_container_scalar_pow(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([4]), 'd': ivy.array([6])}})
    container = container_a ** 2
    assert container['a'] == ivy.array([1])
    assert container.a == ivy.array([1])
    assert container['b']['c'] == ivy.array([4])
    assert container.b.c == ivy.array([4])
    assert container['b']['d'] == ivy.array([9])
    assert container.b.d == ivy.array([9])


def test_container_reverse_scalar_pow(dev_str, call):
    container = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container = 2 ** container
    assert container['a'] == ivy.array([2])
    assert container.a == ivy.array([2])
    assert container['b']['c'] == ivy.array([4])
    assert container.b.c == ivy.array([4])
    assert container['b']['d'] == ivy.array([8])
    assert container.b.d == ivy.array([8])


def test_container_scalar_addition(dev_str, call):
    container = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container += 3
    assert container['a'] == ivy.array([4])
    assert container.a == ivy.array([4])
    assert container['b']['c'] == ivy.array([5])
    assert container.b.c == ivy.array([5])
    assert container['b']['d'] == ivy.array([6])
    assert container.b.d == ivy.array([6])


def test_container_reverse_scalar_addition(dev_str, call):
    container = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container = 3 + container
    assert container['a'] == ivy.array([4])
    assert container.a == ivy.array([4])
    assert container['b']['c'] == ivy.array([5])
    assert container.b.c == ivy.array([5])
    assert container['b']['d'] == ivy.array([6])
    assert container.b.d == ivy.array([6])


def test_container_addition(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([4]), 'd': ivy.array([6])}})
    container = container_a + container_b
    assert container['a'] == ivy.array([3])
    assert container.a == ivy.array([3])
    assert container['b']['c'] == ivy.array([6])
    assert container.b.c == ivy.array([6])
    assert container['b']['d'] == ivy.array([9])
    assert container.b.d == ivy.array([9])


def test_container_scalar_subtraction(dev_str, call):
    container = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container -= 1
    assert container['a'] == ivy.array([0])
    assert container.a == ivy.array([0])
    assert container['b']['c'] == ivy.array([1])
    assert container.b.c == ivy.array([1])
    assert container['b']['d'] == ivy.array([2])
    assert container.b.d == ivy.array([2])


def test_container_reverse_scalar_subtraction(dev_str, call):
    container = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container = 1 - container
    assert container['a'] == ivy.array([0])
    assert container.a == ivy.array([0])
    assert container['b']['c'] == ivy.array([-1])
    assert container.b.c == ivy.array([-1])
    assert container['b']['d'] == ivy.array([-2])
    assert container.b.d == ivy.array([-2])


def test_container_subtraction(dev_str, call):
    container_a = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([4]), 'd': ivy.array([6])}})
    container_b = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([1]), 'd': ivy.array([4])}})
    container = container_a - container_b
    assert container['a'] == ivy.array([1])
    assert container.a == ivy.array([1])
    assert container['b']['c'] == ivy.array([3])
    assert container.b.c == ivy.array([3])
    assert container['b']['d'] == ivy.array([2])
    assert container.b.d == ivy.array([2])


def test_container_sum(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([4]), 'd': ivy.array([6])}})
    container = sum([container_a, container_b])
    assert container['a'] == ivy.array([3])
    assert container.a == ivy.array([3])
    assert container['b']['c'] == ivy.array([6])
    assert container.b.c == ivy.array([6])
    assert container['b']['d'] == ivy.array([9])
    assert container.b.d == ivy.array([9])


def test_container_scalar_multiplication(dev_str, call):
    container = Container({'a': ivy.array([1.]), 'b': {'c': ivy.array([2.]), 'd': ivy.array([3.])}})
    container *= 2.5
    assert container['a'] == ivy.array([2.5])
    assert container.a == ivy.array([2.5])
    assert container['b']['c'] == ivy.array([5.])
    assert container.b.c == ivy.array([5.])
    assert container['b']['d'] == ivy.array([7.5])
    assert container.b.d == ivy.array([7.5])


def test_container_reverse_scalar_multiplication(dev_str, call):
    container = Container({'a': ivy.array([1.]), 'b': {'c': ivy.array([2.]), 'd': ivy.array([3.])}})
    container = 2.5 * container
    assert container['a'] == ivy.array([2.5])
    assert container.a == ivy.array([2.5])
    assert container['b']['c'] == ivy.array([5.])
    assert container.b.c == ivy.array([5.])
    assert container['b']['d'] == ivy.array([7.5])
    assert container.b.d == ivy.array([7.5])


def test_container_multiplication(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([2]), 'd': ivy.array([3])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([4]), 'd': ivy.array([6])}})
    container = container_a * container_b
    assert container['a'] == ivy.array([2])
    assert container.a == ivy.array([2])
    assert container['b']['c'] == ivy.array([8])
    assert container.b.c == ivy.array([8])
    assert container['b']['d'] == ivy.array([18])
    assert container.b.d == ivy.array([18])


def test_container_scalar_truediv(dev_str, call):
    container = Container({'a': ivy.array([1.]), 'b': {'c': ivy.array([5.]), 'd': ivy.array([5.])}})
    container /= 2
    assert container['a'] == ivy.array([0.5])
    assert container.a == ivy.array([0.5])
    assert container['b']['c'] == ivy.array([2.5])
    assert container.b.c == ivy.array([2.5])
    assert container['b']['d'] == ivy.array([2.5])
    assert container.b.d == ivy.array([2.5])


def test_container_reverse_scalar_truediv(dev_str, call):
    container = Container({'a': ivy.array([1.]), 'b': {'c': ivy.array([5.]), 'd': ivy.array([5.])}})
    container = 2 / container
    assert container['a'] == ivy.array([2.])
    assert container.a == ivy.array([2.])
    assert container['b']['c'] == ivy.array([0.4])
    assert container.b.c == ivy.array([0.4])
    assert container['b']['d'] == ivy.array([0.4])
    assert container.b.d == ivy.array([0.4])


def test_container_truediv(dev_str, call):
    container_a = Container({'a': ivy.array([1.]), 'b': {'c': ivy.array([5.]), 'd': ivy.array([5.])}})
    container_b = Container({'a': ivy.array([2.]), 'b': {'c': ivy.array([2.]), 'd': ivy.array([4.])}})
    container = container_a / container_b
    assert container['a'] == ivy.array([0.5])
    assert container.a == ivy.array([0.5])
    assert container['b']['c'] == ivy.array([2.5])
    assert container.b.c == ivy.array([2.5])
    assert container['b']['d'] == ivy.array([1.25])
    assert container.b.d == ivy.array([1.25])


def test_container_scalar_floordiv(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit ivy.floordiv is implemented at some point
        pytest.skip()
    container = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([5]), 'd': ivy.array([5])}})
    container //= 2
    assert container['a'] == ivy.array([0])
    assert container.a == ivy.array([0])
    assert container['b']['c'] == ivy.array([2])
    assert container.b.c == ivy.array([2])
    assert container['b']['d'] == ivy.array([2])
    assert container.b.d == ivy.array([2])


def test_container_reverse_scalar_floordiv(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit ivy.floordiv is implemented at some point
        pytest.skip()
    container = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([1]), 'd': ivy.array([7])}})
    container = 5 // container
    assert container['a'] == ivy.array([2])
    assert container.a == ivy.array([2])
    assert container['b']['c'] == ivy.array([5])
    assert container.b.c == ivy.array([5])
    assert container['b']['d'] == ivy.array([0])
    assert container.b.d == ivy.array([0])


def test_container_floordiv(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit ivy.floordiv is implemented at some point
        pytest.skip()
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([5]), 'd': ivy.array([5])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([2]), 'd': ivy.array([4])}})
    container = container_a // container_b
    assert container['a'] == ivy.array([0])
    assert container.a == ivy.array([0])
    assert container['b']['c'] == ivy.array([2])
    assert container.b.c == ivy.array([2])
    assert container['b']['d'] == ivy.array([1])
    assert container.b.d == ivy.array([1])


def test_container_abs(dev_str, call):
    container = abs(Container({'a': ivy.array([1]), 'b': {'c': ivy.array([-2]), 'd': ivy.array([3])}}))
    assert container['a'] == ivy.array([1])
    assert container.a == ivy.array([1])
    assert container['b']['c'] == ivy.array([2])
    assert container.b.c == ivy.array([2])
    assert container['b']['d'] == ivy.array([3])
    assert container.b.d == ivy.array([3])


def test_container_less_than(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([5]), 'd': ivy.array([5])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([2]), 'd': ivy.array([5])}})
    container = container_a < container_b
    assert container['a'] == ivy.array([True])
    assert container.a == ivy.array([True])
    assert container['b']['c'] == ivy.array([False])
    assert container.b.c == ivy.array([False])
    assert container['b']['d'] == ivy.array([False])
    assert container.b.d == ivy.array([False])


def test_container_less_than_or_equal_to(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([5]), 'd': ivy.array([5])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([2]), 'd': ivy.array([5])}})
    container = container_a <= container_b
    assert container['a'] == ivy.array([True])
    assert container.a == ivy.array([True])
    assert container['b']['c'] == ivy.array([False])
    assert container.b.c == ivy.array([False])
    assert container['b']['d'] == ivy.array([True])
    assert container.b.d == ivy.array([True])


def test_container_not_equal_to(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([5]), 'd': ivy.array([5])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([2]), 'd': ivy.array([5])}})
    container = container_a != container_b
    assert container['a'] == ivy.array([True])
    assert container.a == ivy.array([True])
    assert container['b']['c'] == ivy.array([True])
    assert container.b.c == ivy.array([True])
    assert container['b']['d'] == ivy.array([False])
    assert container.b.d == ivy.array([False])


def test_container_greater_than(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([5]), 'd': ivy.array([5])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([2]), 'd': ivy.array([5])}})
    container = container_a > container_b
    assert container['a'] == ivy.array([False])
    assert container.a == ivy.array([False])
    assert container['b']['c'] == ivy.array([True])
    assert container.b.c == ivy.array([True])
    assert container['b']['d'] == ivy.array([False])
    assert container.b.d == ivy.array([False])


def test_container_greater_than_or_equal_to(dev_str, call):
    container_a = Container({'a': ivy.array([1]), 'b': {'c': ivy.array([5]), 'd': ivy.array([5])}})
    container_b = Container({'a': ivy.array([2]), 'b': {'c': ivy.array([2]), 'd': ivy.array([5])}})
    container = container_a >= container_b
    assert container['a'] == ivy.array([False])
    assert container.a == ivy.array([False])
    assert container['b']['c'] == ivy.array([True])
    assert container.b.c == ivy.array([True])
    assert container['b']['d'] == ivy.array([True])
    assert container.b.d == ivy.array([True])


def test_container_and(dev_str, call):
    container_a = Container({'a': ivy.array([True]), 'b': {'c': ivy.array([True]), 'd': ivy.array([False])}})
    container_b = Container({'a': ivy.array([False]), 'b': {'c': ivy.array([True]), 'd': ivy.array([False])}})
    container = container_a and container_b
    assert container['a'] == ivy.array([False])
    assert container.a == ivy.array([False])
    assert container['b']['c'] == ivy.array([True])
    assert container.b.c == ivy.array([True])
    assert container['b']['d'] == ivy.array([False])
    assert container.b.d == ivy.array([False])


def test_container_or(dev_str, call):
    container_a = Container({'a': ivy.array([True]), 'b': {'c': ivy.array([True]), 'd': ivy.array([False])}})
    container_b = Container({'a': ivy.array([False]), 'b': {'c': ivy.array([True]), 'd': ivy.array([False])}})
    container = container_a or container_b
    assert container['a'] == ivy.array([True])
    assert container.a == ivy.array([True])
    assert container['b']['c'] == ivy.array([True])
    assert container.b.c == ivy.array([True])
    assert container['b']['d'] == ivy.array([False])
    assert container.b.d == ivy.array([False])


def test_container_not(dev_str, call):
    container = ~Container({'a': ivy.array([True]), 'b': {'c': ivy.array([True]), 'd': ivy.array([False])}})
    assert container['a'] == ivy.array([False])
    assert container.a == ivy.array([False])
    assert container['b']['c'] == ivy.array([False])
    assert container.b.c == ivy.array([False])
    assert container['b']['d'] == ivy.array([True])
    assert container.b.d == ivy.array([True])


def test_container_xor(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the ^ operator, can add if explicit ivy.logical_xor is implemented at some point
        pytest.skip()
    container_a = Container({'a': ivy.array([True]), 'b': {'c': ivy.array([True]), 'd': ivy.array([False])}})
    container_b = Container({'a': ivy.array([False]), 'b': {'c': ivy.array([True]), 'd': ivy.array([False])}})
    container = container_a ^ container_b
    assert container['a'] == ivy.array([True])
    assert container.a == ivy.array([True])
    assert container['b']['c'] == ivy.array([False])
    assert container.b.c == ivy.array([False])
    assert container['b']['d'] == ivy.array([False])
    assert container.b.d == ivy.array([False])
