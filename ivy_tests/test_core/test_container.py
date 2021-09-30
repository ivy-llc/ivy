# global
import os
import queue
import pickle
import pytest
import random
import numpy as np
import torch.multiprocessing as multiprocessing

# local
import ivy
import ivy_tests.helpers as helpers
from ivy.core.container import Container


def test_container_list_join(dev_str, call):
    container_0 = Container({'a': [ivy.array([1], dev_str=dev_str)],
                             'b': {'c': [ivy.array([2], dev_str=dev_str)], 'd': [ivy.array([3], dev_str=dev_str)]}})
    container_1 = Container({'a': [ivy.array([4], dev_str=dev_str)],
                             'b': {'c': [ivy.array([5], dev_str=dev_str)], 'd': [ivy.array([6], dev_str=dev_str)]}})
    container_list_joined = ivy.Container.list_join([container_0, container_1])
    assert np.allclose(ivy.to_numpy(container_list_joined['a'][0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_list_joined.a[0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_list_joined['b']['c'][0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_list_joined.b.c[0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_list_joined['b']['d'][0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_list_joined.b.d[0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_list_joined['a'][1]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_list_joined.a[1]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_list_joined['b']['c'][1]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container_list_joined.b.c[1]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container_list_joined['b']['d'][1]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container_list_joined.b.d[1]), np.array([6]))


def test_container_list_stack(dev_str, call):
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'a': ivy.array([4], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container_list_stacked = ivy.Container.list_stack([container_0, container_1], 0)
    assert np.allclose(ivy.to_numpy(container_list_stacked['a'][0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.a[0]), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_list_stacked['b']['c'][0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.b.c[0]), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_list_stacked['b']['d'][0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.b.d[0]), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_list_stacked['a'][1]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.a[1]), np.array([4]))
    assert np.allclose(ivy.to_numpy(container_list_stacked['b']['c'][1]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.b.c[1]), np.array([5]))
    assert np.allclose(ivy.to_numpy(container_list_stacked['b']['d'][1]), np.array([6]))
    assert np.allclose(ivy.to_numpy(container_list_stacked.b.d[1]), np.array([6]))


def test_container_concat(dev_str, call):
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'a': ivy.array([4], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container_concatenated = ivy.Container.concat([container_0, container_1], 0)
    assert np.allclose(ivy.to_numpy(container_concatenated['a']), np.array([1, 4]))
    assert np.allclose(ivy.to_numpy(container_concatenated.a), np.array([1, 4]))
    assert np.allclose(ivy.to_numpy(container_concatenated['b']['c']), np.array([2, 5]))
    assert np.allclose(ivy.to_numpy(container_concatenated.b.c), np.array([2, 5]))
    assert np.allclose(ivy.to_numpy(container_concatenated['b']['d']), np.array([3, 6]))
    assert np.allclose(ivy.to_numpy(container_concatenated.b.d), np.array([3, 6]))


def test_container_stack(dev_str, call):
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'a': ivy.array([4], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container_stacked = ivy.Container.stack([container_0, container_1], 0)
    assert np.allclose(ivy.to_numpy(container_stacked['a']), np.array([[1], [4]]))
    assert np.allclose(ivy.to_numpy(container_stacked.a), np.array([[1], [4]]))
    assert np.allclose(ivy.to_numpy(container_stacked['b']['c']), np.array([[2], [5]]))
    assert np.allclose(ivy.to_numpy(container_stacked.b.c), np.array([[2], [5]]))
    assert np.allclose(ivy.to_numpy(container_stacked['b']['d']), np.array([[3], [6]]))
    assert np.allclose(ivy.to_numpy(container_stacked.b.d), np.array([[3], [6]]))


def test_container_combine(dev_str, call):
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'a': ivy.array([4], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'e': ivy.array([6], dev_str=dev_str)}})
    container_comb = ivy.Container.combine(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_comb.a), np.array([4]))
    assert np.equal(ivy.to_numpy(container_comb.b.c), np.array([5]))
    assert np.equal(ivy.to_numpy(container_comb.b.d), np.array([3]))
    assert np.equal(ivy.to_numpy(container_comb.b.e), np.array([6]))


def test_container_diff(dev_str, call):
    # all different arrays
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'a': ivy.array([4], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.a.diff_1), np.array([4]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_1), np.array([6]))
    container_diff_diff_only = ivy.Container.diff(container_0, container_1, mode='diff_only')
    assert container_diff_diff_only.to_dict() == container_diff.to_dict()
    container_diff_same_only = ivy.Container.diff(container_0, container_1, mode='same_only')
    assert container_diff_same_only.to_dict() == {}

    # some different arrays
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_0), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.c.diff_1), np.array([5]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.diff(container_0, container_1, mode='diff_only')
    assert 'a' not in container_diff_diff_only
    assert 'b' in container_diff_diff_only
    assert 'c' in container_diff_diff_only['b']
    assert 'd' not in container_diff_diff_only['b']
    container_diff_same_only = ivy.Container.diff(container_0, container_1, mode='same_only')
    assert 'a' in container_diff_same_only
    assert 'b' in container_diff_same_only
    assert 'c' not in container_diff_same_only['b']
    assert 'd' in container_diff_same_only['b']

    # all different keys
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'e': ivy.array([1], dev_str=dev_str),
                             'f': {'g': ivy.array([2], dev_str=dev_str), 'h': ivy.array([3], dev_str=dev_str)}})
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a.diff_0), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.diff_0.d), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.e.diff_1), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.g), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.f.diff_1.h), np.array([3]))
    container_diff_diff_only = ivy.Container.diff(container_0, container_1, mode='diff_only')
    assert container_diff_diff_only.to_dict() == container_diff.to_dict()
    container_diff_same_only = ivy.Container.diff(container_0, container_1, mode='same_only')
    assert container_diff_same_only.to_dict() == {}

    # some different keys
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'e': ivy.array([3], dev_str=dev_str)}})
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d.diff_0), np.array([3]))
    assert np.equal(ivy.to_numpy(container_diff.b.e.diff_1), np.array([3]))
    container_diff_diff_only = ivy.Container.diff(container_0, container_1, mode='diff_only')
    assert 'a' not in container_diff_diff_only
    assert 'b' in container_diff_diff_only
    assert 'c' not in container_diff_diff_only['b']
    assert 'd' in container_diff_diff_only['b']
    assert 'e' in container_diff_diff_only['b']
    container_diff_same_only = ivy.Container.diff(container_0, container_1, mode='same_only')
    assert 'a' in container_diff_same_only
    assert 'b' in container_diff_same_only
    assert 'c' in container_diff_same_only['b']
    assert 'd' not in container_diff_same_only['b']
    assert 'e' not in container_diff_same_only['b']

    # same containers
    container_0 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_1 = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_diff = ivy.Container.diff(container_0, container_1)
    assert np.equal(ivy.to_numpy(container_diff.a), np.array([1]))
    assert np.equal(ivy.to_numpy(container_diff.b.c), np.array([2]))
    assert np.equal(ivy.to_numpy(container_diff.b.d), np.array([3]))
    container_diff_diff_only = ivy.Container.diff(container_0, container_1, mode='diff_only')
    assert container_diff_diff_only.to_dict() == {}
    container_diff_same_only = ivy.Container.diff(container_0, container_1, mode='same_only')
    assert container_diff_same_only.to_dict() == container_diff.to_dict()

    # all different strings
    container_0 = Container({'a': '1',
                             'b': {'c': '2', 'd': '3'}})
    container_1 = Container({'a': '4',
                             'b': {'c': '5', 'd': '6'}})
    container_diff = ivy.Container.diff(container_0, container_1)
    assert container_diff.a.diff_0 == '1'
    assert container_diff.a.diff_1 == '4'
    assert container_diff.b.c.diff_0 == '2'
    assert container_diff.b.c.diff_1 == '5'
    assert container_diff.b.d.diff_0 == '3'
    assert container_diff.b.d.diff_1 == '6'
    container_diff_diff_only = ivy.Container.diff(container_0, container_1, mode='diff_only')
    assert container_diff_diff_only.to_dict() == container_diff.to_dict()
    container_diff_same_only = ivy.Container.diff(container_0, container_1, mode='same_only')
    assert container_diff_same_only.to_dict() == {}


def test_container_from_dict(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_from_kwargs(dev_str, call):
    container = Container(a=ivy.array([1], dev_str=dev_str),
                          b={'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)})
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_reduce_sum(dev_str, call):
    dict_in = {'a': ivy.array([1., 2., 3.], dev_str=dev_str),
               'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str), 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reduced_sum = container.reduce_sum()
    assert np.allclose(ivy.to_numpy(container_reduced_sum['a']), np.array([6.]))
    assert np.allclose(ivy.to_numpy(container_reduced_sum.a), np.array([6.]))
    assert np.allclose(ivy.to_numpy(container_reduced_sum['b']['c']), np.array([12.]))
    assert np.allclose(ivy.to_numpy(container_reduced_sum.b.c), np.array([12.]))
    assert np.allclose(ivy.to_numpy(container_reduced_sum['b']['d']), np.array([18.]))
    assert np.allclose(ivy.to_numpy(container_reduced_sum.b.d), np.array([18.]))


def test_container_reduce_prod(dev_str, call):
    dict_in = {'a': ivy.array([1., 2., 3.], dev_str=dev_str),
               'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str), 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reduced_prod = container.reduce_prod()
    assert np.allclose(ivy.to_numpy(container_reduced_prod['a']), np.array([6.]))
    assert np.allclose(ivy.to_numpy(container_reduced_prod.a), np.array([6.]))
    assert np.allclose(ivy.to_numpy(container_reduced_prod['b']['c']), np.array([48.]))
    assert np.allclose(ivy.to_numpy(container_reduced_prod.b.c), np.array([48.]))
    assert np.allclose(ivy.to_numpy(container_reduced_prod['b']['d']), np.array([162.]))
    assert np.allclose(ivy.to_numpy(container_reduced_prod.b.d), np.array([162.]))


def test_container_reduce_mean(dev_str, call):
    dict_in = {'a': ivy.array([1., 2., 3.], dev_str=dev_str),
               'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str), 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reduced_mean = container.reduce_mean()
    assert np.allclose(ivy.to_numpy(container_reduced_mean['a']), np.array([2.]))
    assert np.allclose(ivy.to_numpy(container_reduced_mean.a), np.array([2.]))
    assert np.allclose(ivy.to_numpy(container_reduced_mean['b']['c']), np.array([4.]))
    assert np.allclose(ivy.to_numpy(container_reduced_mean.b.c), np.array([4.]))
    assert np.allclose(ivy.to_numpy(container_reduced_mean['b']['d']), np.array([6.]))
    assert np.allclose(ivy.to_numpy(container_reduced_mean.b.d), np.array([6.]))


def test_container_reduce_var(dev_str, call):
    dict_in = {'a': ivy.array([1., 2., 3.], dev_str=dev_str),
               'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str), 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reduced_var = container.reduce_var()
    assert np.allclose(ivy.to_numpy(container_reduced_var['a']), np.array([2 / 3]))
    assert np.allclose(ivy.to_numpy(container_reduced_var.a), np.array([2 / 3]))
    assert np.allclose(ivy.to_numpy(container_reduced_var['b']['c']), np.array([8 / 3]))
    assert np.allclose(ivy.to_numpy(container_reduced_var.b.c), np.array([8 / 3]))
    assert np.allclose(ivy.to_numpy(container_reduced_var['b']['d']), np.array([6.]))
    assert np.allclose(ivy.to_numpy(container_reduced_var.b.d), np.array([6.]))


def test_container_reduce_std(dev_str, call):
    dict_in = {'a': ivy.array([1., 2., 3.], dev_str=dev_str),
               'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str), 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reduced_std = container.reduce_std()
    assert np.allclose(ivy.to_numpy(container_reduced_std['a']), np.array([2 / 3]) ** 0.5)
    assert np.allclose(ivy.to_numpy(container_reduced_std.a), np.array([2 / 3]) ** 0.5)
    assert np.allclose(ivy.to_numpy(container_reduced_std['b']['c']), np.array([8 / 3]) ** 0.5)
    assert np.allclose(ivy.to_numpy(container_reduced_std.b.c), np.array([8 / 3]) ** 0.5)
    assert np.allclose(ivy.to_numpy(container_reduced_std['b']['d']), np.array([6.]) ** 0.5)
    assert np.allclose(ivy.to_numpy(container_reduced_std.b.d), np.array([6.]) ** 0.5)


def test_container_reduce_min(dev_str, call):
    dict_in = {'a': ivy.array([1., 2., 3.], dev_str=dev_str),
               'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str), 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reduced_min = container.reduce_min()
    assert np.allclose(ivy.to_numpy(container_reduced_min['a']), np.array([1.]))
    assert np.allclose(ivy.to_numpy(container_reduced_min.a), np.array([1.]))
    assert np.allclose(ivy.to_numpy(container_reduced_min['b']['c']), np.array([2.]))
    assert np.allclose(ivy.to_numpy(container_reduced_min.b.c), np.array([2.]))
    assert np.allclose(ivy.to_numpy(container_reduced_min['b']['d']), np.array([3.]))
    assert np.allclose(ivy.to_numpy(container_reduced_min.b.d), np.array([3.]))


def test_container_reduce_max(dev_str, call):
    dict_in = {'a': ivy.array([1., 2., 3.], dev_str=dev_str),
               'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str), 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reduced_max = container.reduce_max()
    assert np.allclose(ivy.to_numpy(container_reduced_max['a']), np.array([3.]))
    assert np.allclose(ivy.to_numpy(container_reduced_max.a), np.array([3.]))
    assert np.allclose(ivy.to_numpy(container_reduced_max['b']['c']), np.array([6.]))
    assert np.allclose(ivy.to_numpy(container_reduced_max.b.c), np.array([6.]))
    assert np.allclose(ivy.to_numpy(container_reduced_max['b']['d']), np.array([9.]))
    assert np.allclose(ivy.to_numpy(container_reduced_max.b.d), np.array([9.]))


def test_container_minimum(dev_str, call):
    container = Container({'a': ivy.array([1., 2., 3.], dev_str=dev_str),
                           'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str),
                                 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}})
    other = Container({'a': ivy.array([2., 3., 2.], dev_str=dev_str),
                       'b': {'c': ivy.array([1., 5., 4.], dev_str=dev_str),
                             'd': ivy.array([4., 7., 8.], dev_str=dev_str)}})

    # against number
    container_minimum = container.minimum(5.)
    assert np.allclose(ivy.to_numpy(container_minimum['a']), np.array([1., 2., 3.]))
    assert np.allclose(ivy.to_numpy(container_minimum.a), np.array([1., 2., 3.]))
    assert np.allclose(ivy.to_numpy(container_minimum['b']['c']), np.array([2., 4., 5.]))
    assert np.allclose(ivy.to_numpy(container_minimum.b.c), np.array([2., 4., 5.]))
    assert np.allclose(ivy.to_numpy(container_minimum['b']['d']), np.array([3., 5., 5.]))
    assert np.allclose(ivy.to_numpy(container_minimum.b.d), np.array([3., 5., 5.]))

    # against container
    container_minimum = container.minimum(other)
    assert np.allclose(ivy.to_numpy(container_minimum['a']), np.array([1., 2., 2.]))
    assert np.allclose(ivy.to_numpy(container_minimum.a), np.array([1., 2., 2.]))
    assert np.allclose(ivy.to_numpy(container_minimum['b']['c']), np.array([1., 4., 4.]))
    assert np.allclose(ivy.to_numpy(container_minimum.b.c), np.array([1., 4., 4.]))
    assert np.allclose(ivy.to_numpy(container_minimum['b']['d']), np.array([3., 6., 8.]))
    assert np.allclose(ivy.to_numpy(container_minimum.b.d), np.array([3., 6., 8.]))


def test_container_maximum(dev_str, call):
    container = Container({'a': ivy.array([1., 2., 3.], dev_str=dev_str),
                           'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str),
                                 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}})
    other = Container({'a': ivy.array([2., 3., 2.], dev_str=dev_str),
                       'b': {'c': ivy.array([1., 5., 4.], dev_str=dev_str),
                             'd': ivy.array([4., 7., 8.], dev_str=dev_str)}})

    # against number
    container_maximum = container.maximum(4.)
    assert np.allclose(ivy.to_numpy(container_maximum['a']), np.array([4., 4., 4.]))
    assert np.allclose(ivy.to_numpy(container_maximum.a), np.array([4., 4., 4.]))
    assert np.allclose(ivy.to_numpy(container_maximum['b']['c']), np.array([4., 4., 6.]))
    assert np.allclose(ivy.to_numpy(container_maximum.b.c), np.array([4., 4., 6.]))
    assert np.allclose(ivy.to_numpy(container_maximum['b']['d']), np.array([4., 6., 9.]))
    assert np.allclose(ivy.to_numpy(container_maximum.b.d), np.array([4., 6., 9.]))

    # against container
    container_maximum = container.maximum(other)
    assert np.allclose(ivy.to_numpy(container_maximum['a']), np.array([2., 3., 3.]))
    assert np.allclose(ivy.to_numpy(container_maximum.a), np.array([2., 3., 3.]))
    assert np.allclose(ivy.to_numpy(container_maximum['b']['c']), np.array([2., 5., 6.]))
    assert np.allclose(ivy.to_numpy(container_maximum.b.c), np.array([2., 5., 6.]))
    assert np.allclose(ivy.to_numpy(container_maximum['b']['d']), np.array([4., 7., 9.]))
    assert np.allclose(ivy.to_numpy(container_maximum.b.d), np.array([4., 7., 9.]))


def test_container_clip(dev_str, call):
    container = Container({'a': ivy.array([1., 2., 3.], dev_str=dev_str),
                           'b': {'c': ivy.array([2., 4., 6.], dev_str=dev_str),
                                 'd': ivy.array([3., 6., 9.], dev_str=dev_str)}})
    container_min = Container({'a': ivy.array([2., 0., 0.], dev_str=dev_str),
                               'b': {'c': ivy.array([0., 5., 0.], dev_str=dev_str),
                                     'd': ivy.array([4., 7., 0.], dev_str=dev_str)}})
    container_max = Container({'a': ivy.array([3., 1., 2.], dev_str=dev_str),
                               'b': {'c': ivy.array([1., 7., 5.], dev_str=dev_str),
                                     'd': ivy.array([5., 8., 8.], dev_str=dev_str)}})

    # against number
    container_clipped = container.clip(2., 6.)
    assert np.allclose(ivy.to_numpy(container_clipped['a']), np.array([2., 2., 3.]))
    assert np.allclose(ivy.to_numpy(container_clipped.a), np.array([2., 2., 3.]))
    assert np.allclose(ivy.to_numpy(container_clipped['b']['c']), np.array([2., 4., 6.]))
    assert np.allclose(ivy.to_numpy(container_clipped.b.c), np.array([2., 4., 6.]))
    assert np.allclose(ivy.to_numpy(container_clipped['b']['d']), np.array([3., 6., 6.]))
    assert np.allclose(ivy.to_numpy(container_clipped.b.d), np.array([3., 6., 6.]))

    if call is helpers.mx_call:
        # MXNet clip does not support arrays for the min and max arguments
        return

    # against container
    container_clipped = container.clip(container_min, container_max)
    assert np.allclose(ivy.to_numpy(container_clipped['a']), np.array([2., 1., 2.]))
    assert np.allclose(ivy.to_numpy(container_clipped.a), np.array([2., 1., 2.]))
    assert np.allclose(ivy.to_numpy(container_clipped['b']['c']), np.array([1., 5., 5.]))
    assert np.allclose(ivy.to_numpy(container_clipped.b.c), np.array([1., 5., 5.]))
    assert np.allclose(ivy.to_numpy(container_clipped['b']['d']), np.array([4., 7., 8.]))
    assert np.allclose(ivy.to_numpy(container_clipped.b.d), np.array([4., 7., 8.]))


def test_container_clip_vector_norm(dev_str, call):
    container = Container({'a': ivy.array([[0.8, 2.2], [1.5, 0.2]], dev_str=dev_str)})
    container_clipped = container.clip_vector_norm(2.5, 2.)
    assert np.allclose(ivy.to_numpy(container_clipped['a']),
                       np.array([[0.71749604, 1.9731141], [1.345305, 0.17937401]]))
    assert np.allclose(ivy.to_numpy(container_clipped.a),
                       np.array([[0.71749604, 1.9731141], [1.345305, 0.17937401]]))


def test_container_einsum(dev_str, call):
    dict_in = {'a': ivy.array([[1., 2.], [3., 4.], [5., 6.]], dev_str=dev_str),
               'b': {'c': ivy.array([[2., 4.], [6., 8.], [10., 12.]], dev_str=dev_str),
                     'd': ivy.array([[-2., -4.], [-6., -8.], [-10., -12.]], dev_str=dev_str)}}
    container = Container(dict_in)
    container_einsummed = container.einsum('ij->i')
    assert np.allclose(ivy.to_numpy(container_einsummed['a']), np.array([3., 7., 11.]))
    assert np.allclose(ivy.to_numpy(container_einsummed.a), np.array([3., 7., 11.]))
    assert np.allclose(ivy.to_numpy(container_einsummed['b']['c']), np.array([6., 14., 22.]))
    assert np.allclose(ivy.to_numpy(container_einsummed.b.c), np.array([6., 14., 22.]))
    assert np.allclose(ivy.to_numpy(container_einsummed['b']['d']), np.array([-6., -14., -22.]))
    assert np.allclose(ivy.to_numpy(container_einsummed.b.d), np.array([-6., -14., -22.]))


def test_container_vector_norm(dev_str, call):
    dict_in = {'a': ivy.array([[1., 2.], [3., 4.], [5., 6.]], dev_str=dev_str),
               'b': {'c': ivy.array([[2., 4.], [6., 8.], [10., 12.]], dev_str=dev_str),
                     'd': ivy.array([[3., 6.], [9., 12.], [15., 18.]], dev_str=dev_str)}}
    container = Container(dict_in)
    container_normed = container.vector_norm(axis=(-1, -2))
    assert np.allclose(ivy.to_numpy(container_normed['a']), 9.5394)
    assert np.allclose(ivy.to_numpy(container_normed.a), 9.5394)
    assert np.allclose(ivy.to_numpy(container_normed['b']['c']), 19.0788)
    assert np.allclose(ivy.to_numpy(container_normed.b.c), 19.0788)
    assert np.allclose(ivy.to_numpy(container_normed['b']['d']), 28.6182)
    assert np.allclose(ivy.to_numpy(container_normed.b.d), 28.6182)


def test_container_matrix_norm(dev_str, call):
    if call is helpers.mx_call:
        # MXNet does not support matrix norm
        pytest.skip()
    dict_in = {'a': ivy.array([[1., 2.], [3., 4.], [5., 6.]], dev_str=dev_str),
               'b': {'c': ivy.array([[2., 4.], [6., 8.], [10., 12.]], dev_str=dev_str),
                     'd': ivy.array([[3., 6.], [9., 12.], [15., 18.]], dev_str=dev_str)}}
    container = Container(dict_in)
    container_normed = container.matrix_norm(axis=(-1, -2))
    assert np.allclose(ivy.to_numpy(container_normed['a']), 9.52551809)
    assert np.allclose(ivy.to_numpy(container_normed.a), 9.52551809)
    assert np.allclose(ivy.to_numpy(container_normed['b']['c']), 19.05103618)
    assert np.allclose(ivy.to_numpy(container_normed.b.c), 19.05103618)
    assert np.allclose(ivy.to_numpy(container_normed['b']['d']), 28.57655427)
    assert np.allclose(ivy.to_numpy(container_normed.b.d), 28.57655427)


def test_container_flip(dev_str, call):
    dict_in = {'a': ivy.array([[1., 2.], [3., 4.], [5., 6.]], dev_str=dev_str),
               'b': {'c': ivy.array([[2., 4.], [6., 8.], [10., 12.]], dev_str=dev_str),
                     'd': ivy.array([[-2., -4.], [-6., -8.], [-10., -12.]], dev_str=dev_str)}}
    container = Container(dict_in)
    container_flipped = container.flip(-1)
    assert np.allclose(ivy.to_numpy(container_flipped['a']), np.array([[2., 1.], [4., 3.], [6., 5.]]))
    assert np.allclose(ivy.to_numpy(container_flipped.a), np.array([[2., 1.], [4., 3.], [6., 5.]]))
    assert np.allclose(ivy.to_numpy(container_flipped['b']['c']), np.array([[4., 2.], [8., 6.], [12., 10.]]))
    assert np.allclose(ivy.to_numpy(container_flipped.b.c), np.array([[4., 2.], [8., 6.], [12., 10.]]))
    assert np.allclose(ivy.to_numpy(container_flipped['b']['d']), np.array([[-4., -2.], [-8., -6.], [-12., -10.]]))
    assert np.allclose(ivy.to_numpy(container_flipped.b.d), np.array([[-4., -2.], [-8., -6.], [-12., -10.]]))


def test_container_as_ones(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)

    container_ones = container.as_ones()
    assert np.allclose(ivy.to_numpy(container_ones['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_ones.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_ones['b']['c']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_ones.b.c), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_ones['b']['d']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_ones.b.d), np.array([1]))


def test_container_as_zeros(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)

    container_zeros = container.as_zeros()
    assert np.allclose(ivy.to_numpy(container_zeros['a']), np.array([0]))
    assert np.allclose(ivy.to_numpy(container_zeros.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container_zeros['b']['c']), np.array([0]))
    assert np.allclose(ivy.to_numpy(container_zeros.b.c), np.array([0]))
    assert np.allclose(ivy.to_numpy(container_zeros['b']['d']), np.array([0]))
    assert np.allclose(ivy.to_numpy(container_zeros.b.d), np.array([0]))


def test_container_as_bools(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': [], 'd': True}}
    container = Container(dict_in)

    container_bools = container.as_bools()
    assert container_bools['a'] is True
    assert container_bools.a is True
    assert container_bools['b']['c'] is False
    assert container_bools.b.c is False
    assert container_bools['b']['d'] is True
    assert container_bools.b.d is True


def test_container_all_true(dev_str, call):
    assert not Container({'a': ivy.array([1], dev_str=dev_str), 'b': {'c': [], 'd': True}}).all_true()
    assert Container({'a': ivy.array([1], dev_str=dev_str), 'b': {'c': [1], 'd': True}}).all_true()
    # noinspection PyBroadException
    try:
        assert Container({'a': ivy.array([1], dev_str=dev_str), 'b': {'c': [1], 'd': True}}).all_true(
            assert_is_bool=True)
        error_raised = False
    except AssertionError:
        error_raised = True
    assert error_raised


def test_container_all_false(dev_str, call):
    assert Container({'a': False, 'b': {'c': [], 'd': 0}}).all_false()
    assert not Container({'a': False, 'b': {'c': [1], 'd': 0}}).all_false()
    # noinspection PyBroadException
    try:
        assert Container({'a': ivy.array([1], dev_str=dev_str), 'b': {'c': [1], 'd': True}}).all_false(
            assert_is_bool=True)
        error_raised = False
    except AssertionError:
        error_raised = True
    assert error_raised


def test_container_as_random_uniform(dev_str, call):
    dict_in = {'a': ivy.array([1.], dev_str=dev_str),
               'b': {'c': ivy.array([2.], dev_str=dev_str), 'd': ivy.array([3.], dev_str=dev_str)}}
    container = Container(dict_in)

    container_random = container.as_random_uniform()
    assert (ivy.to_numpy(container_random['a']) != np.array([1.]))[0]
    assert (ivy.to_numpy(container_random.a) != np.array([1.]))[0]
    assert (ivy.to_numpy(container_random['b']['c']) != np.array([2.]))[0]
    assert (ivy.to_numpy(container_random.b.c) != np.array([2.]))[0]
    assert (ivy.to_numpy(container_random['b']['d']) != np.array([3.]))[0]
    assert (ivy.to_numpy(container_random.b.d) != np.array([3.]))[0]


def test_container_expand_dims(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)

    # without key_chains specification
    container_expanded_dims = container.expand_dims(0)
    assert np.allclose(ivy.to_numpy(container_expanded_dims['a']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims['b']['c']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.b.c), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims['b']['d']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.b.d), np.array([[3]]))

    # with key_chains to apply
    container_expanded_dims = container.expand_dims(0, ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_expanded_dims['a']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims['b']['c']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.b.c), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.b.d), np.array([3]))

    # with key_chains to apply pruned
    container_expanded_dims = container.expand_dims(0, ['a', 'b/c'], prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_expanded_dims['a']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims['b']['c']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.b.c), np.array([[2]]))
    assert 'b/d' not in container_expanded_dims

    # with key_chains to not apply
    container_expanded_dims = container.expand_dims(0, Container({'a': None, 'b': {'d': None}}), to_apply=False)
    assert np.allclose(ivy.to_numpy(container_expanded_dims['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims['b']['c']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.b.c), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.b.d), np.array([3]))

    # with key_chains to not apply pruned
    container_expanded_dims = container.expand_dims(0, Container({'a': None, 'b': {'d': None}}), to_apply=False,
                                                    prune_unapplied=True)
    assert 'a' not in container_expanded_dims
    assert np.allclose(ivy.to_numpy(container_expanded_dims['b']['c']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_expanded_dims.b.c), np.array([[2]]))
    assert 'b/d' not in container_expanded_dims


def test_container_unstack(dev_str, call):
    dict_in = {'a': ivy.array([[1], [2], [3]], dev_str=dev_str),
               'b': {'c': ivy.array([[2], [3], [4]], dev_str=dev_str),
                     'd': ivy.array([[3], [4], [5]], dev_str=dev_str)}}
    container = Container(dict_in)

    # without key_chains specification
    container_unstacked = container.unstack(0)
    for cont, a, bc, bd in zip(container_unstacked, [1, 2, 3], [2, 3, 4], [3, 4, 5]):
        assert np.array_equal(ivy.to_numpy(cont['a']), np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont.a), np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont['b']['c']), np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont.b.c), np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont['b']['d']), np.array([bd]))
        assert np.array_equal(ivy.to_numpy(cont.b.d), np.array([bd]))


def test_container_split(dev_str, call):
    dict_in = {'a': ivy.array([[1], [2], [3]], dev_str=dev_str),
               'b': {'c': ivy.array([[2], [3], [4]], dev_str=dev_str),
                     'd': ivy.array([[3], [4], [5]], dev_str=dev_str)}}
    container = Container(dict_in)

    # without key_chains specification
    container_split = container.split(1, -1)
    for cont, a, bc, bd in zip(container_split, [1, 2, 3], [2, 3, 4], [3, 4, 5]):
        assert np.array_equal(ivy.to_numpy(cont['a'])[0], np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont.a)[0], np.array([a]))
        assert np.array_equal(ivy.to_numpy(cont['b']['c'])[0], np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont.b.c)[0], np.array([bc]))
        assert np.array_equal(ivy.to_numpy(cont['b']['d'])[0], np.array([bd]))
        assert np.array_equal(ivy.to_numpy(cont.b.d)[0], np.array([bd]))


def test_container_gather(dev_str, call):
    dict_in = {'a': ivy.array([1, 2, 3, 4, 5, 6], dev_str=dev_str),
               'b': {'c': ivy.array([2, 3, 4, 5], dev_str=dev_str), 'd': ivy.array([10, 9, 8, 7, 6], dev_str=dev_str)}}
    container = Container(dict_in)

    # without key_chains specification
    container_gathered = container.gather(ivy.array([1, 3], dev_str=dev_str))
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([9, 7]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([9, 7]))

    # with key_chains to apply
    container_gathered = container.gather(ivy.array([1, 3], dev_str=dev_str), -1, ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([10, 9, 8, 7, 6]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([10, 9, 8, 7, 6]))

    # with key_chains to apply pruned
    container_gathered = container.gather(ivy.array([1, 3], dev_str=dev_str), -1, ['a', 'b/c'], prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([2, 4]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert 'b/d' not in container_gathered

    # with key_chains to not apply
    container_gathered = container.gather(ivy.array([1, 3], dev_str=dev_str), -1,
                                          Container({'a': None, 'b': {'d': None}}),
                                          to_apply=False)
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([1, 2, 3, 4, 5, 6]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([1, 2, 3, 4, 5, 6]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([10, 9, 8, 7, 6]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([10, 9, 8, 7, 6]))

    # with key_chains to not apply pruned
    container_gathered = container.gather(ivy.array([1, 3], dev_str=dev_str), -1,
                                          Container({'a': None, 'b': {'d': None}}),
                                          to_apply=False, prune_unapplied=True)
    assert 'a' not in container_gathered
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([3, 5]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([3, 5]))
    assert 'b/d' not in container_gathered


def test_container_gather_nd(dev_str, call):
    dict_in = {'a': ivy.array([[[1, 2], [3, 4]],
                               [[5, 6], [7, 8]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[8, 7], [6, 5]],
                                     [[4, 3], [2, 1]]], dev_str=dev_str),
                     'd': ivy.array([[[2, 4], [6, 8]],
                                     [[10, 12], [14, 16]]], dev_str=dev_str)}}
    container = Container(dict_in)

    # without key_chains specification
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]], dev_str=dev_str))
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([[6, 8], [10, 12]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([[6, 8], [10, 12]]))

    # with key_chains to apply
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]], dev_str=dev_str), ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['d']), np.array([[[2, 4], [6, 8]],
                                                                             [[10, 12], [14, 16]]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.d), np.array([[[2, 4], [6, 8]],
                                                                       [[10, 12], [14, 16]]]))

    # with key_chains to apply pruned
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]], dev_str=dev_str), ['a', 'b/c'],
                                             prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_gathered['a']), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered.a), np.array([[3, 4], [5, 6]]))
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert 'b/d' not in container_gathered

    # with key_chains to not apply
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]], dev_str=dev_str),
                                             Container({'a': None, 'b': {'d': None}}),
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
    container_gathered = container.gather_nd(ivy.array([[0, 1], [1, 0]], dev_str=dev_str),
                                             Container({'a': None, 'b': {'d': None}}),
                                             to_apply=False, prune_unapplied=True)
    assert 'a' not in container_gathered
    assert np.allclose(ivy.to_numpy(container_gathered['b']['c']), np.array([[6, 5], [4, 3]]))
    assert np.allclose(ivy.to_numpy(container_gathered.b.c), np.array([[6, 5], [4, 3]]))
    assert 'b/d' not in container_gathered


def test_container_repeat(dev_str, call):
    if call is helpers.mx_call:
        # MXNet does not support repeats specified as array
        pytest.skip()
    dict_in = {'a': ivy.array([[0., 1., 2., 3.]], dev_str=dev_str),
               'b': {'c': ivy.array([[5., 10., 15., 20.]], dev_str=dev_str),
                     'd': ivy.array([[10., 9., 8., 7.]], dev_str=dev_str)}}
    container = Container(dict_in)

    # without key_chains specification
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3], dev_str=dev_str), -1)
    assert np.allclose(ivy.to_numpy(container_repeated['a']), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['d']), np.array([[10., 10., 9., 7., 7., 7.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.d), np.array([[10., 10., 9., 7., 7., 7.]]))

    # with key_chains to apply
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3], dev_str=dev_str), -1, ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_repeated['a']), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['d']), np.array([[10., 9., 8., 7.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.d), np.array([[10., 9., 8., 7.]]))

    # with key_chains to apply pruned
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3], dev_str=dev_str), -1, ['a', 'b/c'],
                                          prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_repeated['a']), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a), np.array([[0., 0., 1., 3., 3., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert 'b/d' not in container_repeated

    # with key_chains to not apply
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3], dev_str=dev_str), -1,
                                          Container({'a': None, 'b': {'d': None}}),
                                          to_apply=False)
    assert np.allclose(ivy.to_numpy(container_repeated['a']), np.array([[0., 1., 2., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a), np.array([[0., 1., 2., 3.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['d']), np.array([[10., 9., 8., 7.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.d), np.array([[10., 9., 8., 7.]]))

    # with key_chains to not apply pruned
    container_repeated = container.repeat(ivy.array([2, 1, 0, 3], dev_str=dev_str), -1,
                                          Container({'a': None, 'b': {'d': None}}),
                                          to_apply=False, prune_unapplied=True)
    assert 'a' not in container_repeated
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c), np.array([[5., 5., 10., 20., 20., 20.]]))
    assert 'b/d' not in container_repeated


def test_container_swapaxes(dev_str, call):
    if call is helpers.mx_call:
        # MXNet does not support repeats specified as array
        pytest.skip()
    dict_in = {'a': ivy.array([[0., 1., 2., 3.]], dev_str=dev_str),
               'b': {'c': ivy.array([[5., 10., 15., 20.]], dev_str=dev_str),
                     'd': ivy.array([[10., 9., 8., 7.]], dev_str=dev_str)}}
    container = Container(dict_in)

    # without key_chains specification
    container_swapped = container.swapaxes(0, 1)
    assert np.allclose(ivy.to_numpy(container_swapped['a']), np.array([[0.], [1.], [2.], [3.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.a), np.array([[0.], [1.], [2.], [3.]]))
    assert np.allclose(ivy.to_numpy(container_swapped['b']['c']), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.b.c), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_swapped['b']['d']), np.array([[10.], [9.], [8.], [7.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.b.d), np.array([[10.], [9.], [8.], [7.]]))

    # with key_chains to apply
    container_swapped = container.swapaxes(0, 1, ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_swapped['a']), np.array([[0.], [1.], [2.], [3.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.a), np.array([[0.], [1.], [2.], [3.]]))
    assert np.allclose(ivy.to_numpy(container_swapped['b']['c']), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.b.c), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_swapped['b']['d']), np.array([10., 9., 8., 7.]))
    assert np.allclose(ivy.to_numpy(container_swapped.b.d), np.array([10., 9., 8., 7.]))

    # with key_chains to apply pruned
    container_swapped = container.swapaxes(0, 1, ['a', 'b/c'], prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_swapped['a']), np.array([[0.], [1.], [2.], [3.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.a), np.array([[0.], [1.], [2.], [3.]]))
    assert np.allclose(ivy.to_numpy(container_swapped['b']['c']), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.b.c), np.array([[5.], [10.], [15.], [20.]]))
    assert 'b/d' not in container_swapped

    # with key_chains to not apply
    container_swapped = container.swapaxes(0, 1, Container({'a': None, 'b': {'d': None}}), to_apply=False)
    assert np.allclose(ivy.to_numpy(container_swapped['a']), np.array([0., 1., 2., 3.]))
    assert np.allclose(ivy.to_numpy(container_swapped.a), np.array([0., 1., 2., 3.]))
    assert np.allclose(ivy.to_numpy(container_swapped['b']['c']), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.b.c), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_swapped['b']['d']), np.array([10., 9., 8., 7.]))
    assert np.allclose(ivy.to_numpy(container_swapped.b.d), np.array([10., 9., 8., 7.]))

    # with key_chains to not apply pruned
    container_swapped = container.swapaxes(0, 1, Container({'a': None, 'b': {'d': None}}), to_apply=False,
                                           prune_unapplied=True)
    assert 'a' not in container_swapped
    assert np.allclose(ivy.to_numpy(container_swapped['b']['c']), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_swapped.b.c), np.array([[5.], [10.], [15.], [20.]]))
    assert 'b/d' not in container_swapped


def test_container_reshape(dev_str, call):
    dict_in = {'a': ivy.array([[0., 1., 2., 3.]], dev_str=dev_str),
               'b': {'c': ivy.array([[5., 10., 15., 20.]], dev_str=dev_str),
                     'd': ivy.array([[10., 9., 8., 7.]], dev_str=dev_str)}}
    container = Container(dict_in)

    # pre_shape only
    container_reshaped = container.reshape((1, 2, 2))
    assert np.allclose(ivy.to_numpy(container_reshaped['a']), np.array([[0., 1.], [2., 3.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.a), np.array([[0., 1.], [2., 3.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped['b']['c']), np.array([[5., 10.], [15., 20.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.b.c), np.array([[5., 10.], [15., 20.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped['b']['d']), np.array([[10., 9.], [8., 7.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.b.d), np.array([[10., 9.], [8., 7.]]))

    # pre_shape and slice
    dict_in = {'a': ivy.array([[[0., 1., 2., 3.], [0., 1., 2., 3.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[5., 10., 15.], [20., 25., 30.]]], dev_str=dev_str),
                     'd': ivy.array([[[10.], [9.]]], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reshaped = container.reshape((-1,), slice(2, None))
    assert np.allclose(ivy.to_numpy(container_reshaped['a']), np.array([[0., 1., 2., 3.], [0., 1., 2., 3.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.a), np.array([[0., 1., 2., 3.], [0., 1., 2., 3.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped['b']['c']), np.array([[5., 10., 15.], [20., 25., 30.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.b.c), np.array([[5., 10., 15.], [20., 25., 30.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped['b']['d']), np.array([[10.], [9.]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.b.d), np.array([[10.], [9.]]))

    # pre_shape, slice and post_shape
    dict_in = {'a': ivy.array([[[0., 1., 2., 3.], [0., 1., 2., 3.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[5., 10., 15.], [20., 25., 30.]]], dev_str=dev_str),
                     'd': ivy.array([[[10.], [9.]]], dev_str=dev_str)}}
    container = Container(dict_in)
    container_reshaped = container.reshape((-1,), slice(2, None), (1,))
    assert np.allclose(ivy.to_numpy(container_reshaped['a']), np.array([[[0.], [1.], [2.], [3.]],
                                                                        [[0.], [1.], [2.], [3.]]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.a), np.array([[[0.], [1.], [2.], [3.]],
                                                                     [[0.], [1.], [2.], [3.]]]))
    assert np.allclose(ivy.to_numpy(container_reshaped['b']['c']), np.array([[[5.], [10.], [15.]],
                                                                             [[20.], [25.], [30.]]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.b.c), np.array([[[5.], [10.], [15.]],
                                                                       [[20.], [25.], [30.]]]))
    assert np.allclose(ivy.to_numpy(container_reshaped['b']['d']), np.array([[[10.]], [[9.]]]))
    assert np.allclose(ivy.to_numpy(container_reshaped.b.d), np.array([[[10.]], [[9.]]]))


def test_container_einops_rearrange(dev_str, call):
    dict_in = {'a': ivy.array([[0., 1., 2., 3.]], dev_str=dev_str),
               'b': {'c': ivy.array([[5., 10., 15., 20.]], dev_str=dev_str),
                     'd': ivy.array([[10., 9., 8., 7.]], dev_str=dev_str)}}
    container = Container(dict_in)

    container_rearranged = container.einops_rearrange('b n -> n b')
    assert np.allclose(ivy.to_numpy(container_rearranged['a']), np.array([[0.], [1.], [2.], [3.]]))
    assert np.allclose(ivy.to_numpy(container_rearranged.a), np.array([[0.], [1.], [2.], [3.]]))
    assert np.allclose(ivy.to_numpy(container_rearranged['b']['c']), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_rearranged.b.c), np.array([[5.], [10.], [15.], [20.]]))
    assert np.allclose(ivy.to_numpy(container_rearranged['b']['d']), np.array([[10.], [9.], [8.], [7.]]))
    assert np.allclose(ivy.to_numpy(container_rearranged.b.d), np.array([[10.], [9.], [8.], [7.]]))


def test_container_einops_reduce(dev_str, call):
    dict_in = {'a': ivy.array([[0., 1., 2., 3.]], dev_str=dev_str),
               'b': {'c': ivy.array([[5., 10., 15., 20.]], dev_str=dev_str),
                     'd': ivy.array([[10., 9., 8., 7.]], dev_str=dev_str)}}
    container = Container(dict_in)

    container_reduced = container.einops_reduce('b n -> b', 'mean')
    assert np.allclose(ivy.to_numpy(container_reduced['a']), np.array([1.5]))
    assert np.allclose(ivy.to_numpy(container_reduced.a), np.array([1.5]))
    assert np.allclose(ivy.to_numpy(container_reduced['b']['c']), np.array([12.5]))
    assert np.allclose(ivy.to_numpy(container_reduced.b.c), np.array([12.5]))
    assert np.allclose(ivy.to_numpy(container_reduced['b']['d']), np.array([8.5]))
    assert np.allclose(ivy.to_numpy(container_reduced.b.d), np.array([8.5]))


def test_container_einops_repeat(dev_str, call):
    dict_in = {'a': ivy.array([[0., 1., 2., 3.]], dev_str=dev_str),
               'b': {'c': ivy.array([[5., 10., 15., 20.]], dev_str=dev_str),
                     'd': ivy.array([[10., 9., 8., 7.]], dev_str=dev_str)}}
    container = Container(dict_in)

    container_repeated = container.einops_repeat('b n -> b n c', c=2)
    assert np.allclose(ivy.to_numpy(container_repeated['a']),
                       np.array([[[0., 0.], [1., 1.], [2., 2.], [3., 3.]]]))
    assert np.allclose(ivy.to_numpy(container_repeated.a),
                       np.array([[[0., 0.], [1., 1.], [2., 2.], [3., 3.]]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['c']),
                       np.array([[[5., 5.], [10., 10.], [15., 15.], [20., 20.]]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.c),
                       np.array([[[5., 5.], [10., 10.], [15., 15.], [20., 20.]]]))
    assert np.allclose(ivy.to_numpy(container_repeated['b']['d']),
                       np.array([[[10., 10.], [9., 9.], [8., 8.], [7., 7.]]]))
    assert np.allclose(ivy.to_numpy(container_repeated.b.d),
                       np.array([[[10., 10.], [9., 9.], [8., 8.], [7., 7.]]]))


def test_container_to_dev(dev_str, call):
    dict_in = {'a': ivy.array([[0., 1., 2., 3.]], dev_str=dev_str),
               'b': {'c': ivy.array([[5., 10., 15., 20.]], dev_str=dev_str),
                     'd': ivy.array([[10., 9., 8., 7.]], dev_str=dev_str)}}
    container = Container(dict_in)

    container_to_cpu = container.to_dev(dev_str)
    assert ivy.dev_str(container_to_cpu['a']) == dev_str
    assert ivy.dev_str(container_to_cpu.a) == dev_str
    assert ivy.dev_str(container_to_cpu['b']['c']) == dev_str
    assert ivy.dev_str(container_to_cpu.b.c) == dev_str
    assert ivy.dev_str(container_to_cpu['b']['d']) == dev_str
    assert ivy.dev_str(container_to_cpu.b.d) == dev_str


def test_container_stop_gradients(dev_str, call):
    dict_in = {'a': ivy.variable(ivy.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dev_str=dev_str)),
               'b': {'c': ivy.variable(ivy.array([[[8., 7.], [6., 5.]], [[4., 3.], [2., 1.]]], dev_str=dev_str)),
                     'd': ivy.variable(ivy.array([[[2., 4.], [6., 8.]], [[10., 12.], [14., 16.]]], dev_str=dev_str))}}
    container = Container(dict_in)
    if call is not helpers.np_call:
        # Numpy does not support variables or gradients
        assert ivy.is_variable(container['a'])
        assert ivy.is_variable(container.a)
        assert ivy.is_variable(container['b']['c'])
        assert ivy.is_variable(container.b.c)
        assert ivy.is_variable(container['b']['d'])
        assert ivy.is_variable(container.b.d)

    # without key_chains specification
    container_stopped_grads = container.stop_gradients()
    assert ivy.is_array(container_stopped_grads['a'])
    assert ivy.is_array(container_stopped_grads.a)
    assert ivy.is_array(container_stopped_grads['b']['c'])
    assert ivy.is_array(container_stopped_grads.b.c)
    assert ivy.is_array(container_stopped_grads['b']['d'])
    assert ivy.is_array(container_stopped_grads.b.d)

    # with key_chains to apply
    container_stopped_grads = container.stop_gradients(key_chains=['a', 'b/c'])
    assert ivy.is_array(container_stopped_grads['a'])
    assert ivy.is_array(container_stopped_grads.a)
    assert ivy.is_array(container_stopped_grads['b']['c'])
    assert ivy.is_array(container_stopped_grads.b.c)
    if call is not helpers.np_call:
        # Numpy does not support variables or gradients
        assert ivy.is_variable(container_stopped_grads['b']['d'])
        assert ivy.is_variable(container_stopped_grads.b.d)

    # with key_chains to apply pruned
    container_stopped_grads = container.stop_gradients(key_chains=['a', 'b/c'], prune_unapplied=True)
    assert ivy.is_array(container_stopped_grads['a'])
    assert ivy.is_array(container_stopped_grads.a)
    assert ivy.is_array(container_stopped_grads['b']['c'])
    assert ivy.is_array(container_stopped_grads.b.c)
    assert 'b/d' not in container_stopped_grads

    # with key_chains to not apply
    container_stopped_grads = container.stop_gradients(key_chains=Container({'a': None, 'b': {'d': None}}),
                                                       to_apply=False)
    if call is not helpers.np_call:
        # Numpy does not support variables or gradients
        assert ivy.is_variable(container_stopped_grads['a'])
        assert ivy.is_variable(container_stopped_grads.a)
    assert ivy.is_array(container_stopped_grads['b']['c'])
    assert ivy.is_array(container_stopped_grads.b.c)
    if call is not helpers.np_call:
        # Numpy does not support variables or gradients
        assert ivy.is_variable(container_stopped_grads['b']['d'])
        assert ivy.is_variable(container_stopped_grads.b.d)

    # with key_chains to not apply pruned
    container_stopped_grads = container.stop_gradients(key_chains=Container({'a': None, 'b': {'d': None}}),
                                                       to_apply=False, prune_unapplied=True)
    assert 'a' not in container_stopped_grads
    assert ivy.is_array(container_stopped_grads['b']['c'])
    assert ivy.is_array(container_stopped_grads.b.c)
    assert 'b/d' not in container_stopped_grads


def test_container_as_variables(dev_str, call):
    dict_in = {'a': ivy.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[8., 7.], [6., 5.]], [[4., 3.], [2., 1.]]], dev_str=dev_str),
                     'd': ivy.array([[[2., 4.], [6., 8.]], [[10., 12.], [14., 16.]]], dev_str=dev_str)}}
    container = Container(dict_in)

    assert ivy.is_array(container['a'])
    assert ivy.is_array(container.a)
    assert ivy.is_array(container['b']['c'])
    assert ivy.is_array(container.b.c)
    assert ivy.is_array(container['b']['d'])
    assert ivy.is_array(container.b.d)

    variable_cont = container.as_variables()

    if call is not helpers.np_call:
        # Numpy does not support variables or gradients
        assert ivy.is_variable(variable_cont['a'])
        assert ivy.is_variable(variable_cont.a)
        assert ivy.is_variable(variable_cont['b']['c'])
        assert ivy.is_variable(variable_cont.b.c)
        assert ivy.is_variable(variable_cont['b']['d'])
        assert ivy.is_variable(variable_cont.b.d)


def test_container_as_arrays(dev_str, call):
    dict_in = {'a': ivy.variable(ivy.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dev_str=dev_str)),
               'b': {'c': ivy.variable(ivy.array([[[8., 7.], [6., 5.]], [[4., 3.], [2., 1.]]], dev_str=dev_str)),
                     'd': ivy.variable(ivy.array([[[2., 4.], [6., 8.]], [[10., 12.], [14., 16.]]], dev_str=dev_str))}}
    container = Container(dict_in)
    if call is not helpers.np_call:
        # Numpy does not support variables or gradients
        assert ivy.is_variable(container['a'])
        assert ivy.is_variable(container.a)
        assert ivy.is_variable(container['b']['c'])
        assert ivy.is_variable(container.b.c)
        assert ivy.is_variable(container['b']['d'])
        assert ivy.is_variable(container.b.d)

    # without key_chains specification
    container_as_arrays = container.as_arrays()
    assert ivy.is_array(container_as_arrays['a'])
    assert ivy.is_array(container_as_arrays.a)
    assert ivy.is_array(container_as_arrays['b']['c'])
    assert ivy.is_array(container_as_arrays.b.c)
    assert ivy.is_array(container_as_arrays['b']['d'])
    assert ivy.is_array(container_as_arrays.b.d)


def test_container_to_numpy(dev_str, call):
    dict_in = {'a': ivy.variable(ivy.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dev_str=dev_str)),
               'b': {'c': ivy.variable(ivy.array([[[8., 7.], [6., 5.]], [[4., 3.], [2., 1.]]], dev_str=dev_str)),
                     'd': ivy.variable(ivy.array([[[2., 4.], [6., 8.]], [[10., 12.], [14., 16.]]], dev_str=dev_str))}}
    container = Container(dict_in)
    assert ivy.is_array(container['a'])
    assert ivy.is_array(container.a)
    assert ivy.is_array(container['b']['c'])
    assert ivy.is_array(container.b.c)
    assert ivy.is_array(container['b']['d'])
    assert ivy.is_array(container.b.d)

    # without key_chains specification
    container_to_numpy = container.to_numpy()
    assert isinstance(container_to_numpy['a'], np.ndarray)
    assert isinstance(container_to_numpy.a, np.ndarray)
    assert isinstance(container_to_numpy['b']['c'], np.ndarray)
    assert isinstance(container_to_numpy.b.c, np.ndarray)
    assert isinstance(container_to_numpy['b']['d'], np.ndarray)
    assert isinstance(container_to_numpy.b.d, np.ndarray)


def test_container_arrays_as_lists(dev_str, call):
    dict_in = {'a': ivy.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[8., 7.], [6., 5.]], [[4., 3.], [2., 1.]]], dev_str=dev_str),
                     'd': ivy.array([[[2., 4.], [6., 8.]], [[10., 12.], [14., 16.]]], dev_str=dev_str)}}
    container = Container(dict_in)

    assert ivy.is_array(container['a'])
    assert ivy.is_array(container.a)
    assert ivy.is_array(container['b']['c'])
    assert ivy.is_array(container.b.c)
    assert ivy.is_array(container['b']['d'])
    assert ivy.is_array(container.b.d)

    # without key_chains specification
    container_arrays_as_lists = container.arrays_as_lists()
    assert isinstance(container_arrays_as_lists['a'], list)
    assert isinstance(container_arrays_as_lists.a, list)
    assert isinstance(container_arrays_as_lists['b']['c'], list)
    assert isinstance(container_arrays_as_lists.b.c, list)
    assert isinstance(container_arrays_as_lists['b']['d'], list)
    assert isinstance(container_arrays_as_lists.b.d, list)


def test_container_has_key(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    assert container.has_key('a')
    assert container.has_key('b')
    assert container.has_key('c')
    assert container.has_key('d')
    assert not container.has_key('e')
    assert not container.has_key('f')


def test_container_has_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    assert container.has_key_chain('a')
    assert container.has_key_chain('b')
    assert container.has_key_chain('b/c')
    assert container.has_key_chain('b/d')
    assert not container.has_key_chain('b/e')
    assert not container.has_key_chain('c')


def test_container_has_nans(dev_str, call):
    container = Container({'a': ivy.array([1., 2.], dev_str=dev_str),
                           'b': {'c': ivy.array([2., 3.], dev_str=dev_str), 'd': ivy.array([3., 4.], dev_str=dev_str)}})
    container_nan = Container({'a': ivy.array([1., 2.], dev_str=dev_str),
                               'b': {'c': ivy.array([float('nan'), 3.], dev_str=dev_str),
                                     'd': ivy.array([3., 4.], dev_str=dev_str)}})
    container_inf = Container({'a': ivy.array([1., 2.], dev_str=dev_str),
                               'b': {'c': ivy.array([2., 3.], dev_str=dev_str),
                                     'd': ivy.array([3., float('inf')], dev_str=dev_str)}})
    container_nan_n_inf = Container({'a': ivy.array([1., 2.], dev_str=dev_str),
                                     'b': {'c': ivy.array([float('nan'), 3.], dev_str=dev_str),
                                           'd': ivy.array([3., float('inf')], dev_str=dev_str)}})

    # global

    # with inf check
    assert not container.has_nans()
    assert container_nan.has_nans()
    assert container_inf.has_nans()
    assert container_nan_n_inf.has_nans()

    # without inf check
    assert not container.has_nans(include_infs=False)
    assert container_nan.has_nans(include_infs=False)
    assert not container_inf.has_nans(include_infs=False)
    assert container_nan_n_inf.has_nans(include_infs=False)

    # leafwise

    # with inf check
    container_hn = container.has_nans(leafwise=True)
    assert container_hn.a is False
    assert container_hn.b.c is False
    assert container_hn.b.d is False

    container_nan_hn = container_nan.has_nans(leafwise=True)
    assert container_nan_hn.a is False
    assert container_nan_hn.b.c is True
    assert container_nan_hn.b.d is False

    container_inf_hn = container_inf.has_nans(leafwise=True)
    assert container_inf_hn.a is False
    assert container_inf_hn.b.c is False
    assert container_inf_hn.b.d is True

    container_nan_n_inf_hn = container_nan_n_inf.has_nans(leafwise=True)
    assert container_nan_n_inf_hn.a is False
    assert container_nan_n_inf_hn.b.c is True
    assert container_nan_n_inf_hn.b.d is True

    # without inf check
    container_hn = container.has_nans(leafwise=True, include_infs=False)
    assert container_hn.a is False
    assert container_hn.b.c is False
    assert container_hn.b.d is False

    container_nan_hn = container_nan.has_nans(leafwise=True, include_infs=False)
    assert container_nan_hn.a is False
    assert container_nan_hn.b.c is True
    assert container_nan_hn.b.d is False

    container_inf_hn = container_inf.has_nans(leafwise=True, include_infs=False)
    assert container_inf_hn.a is False
    assert container_inf_hn.b.c is False
    assert container_inf_hn.b.d is False

    container_nan_n_inf_hn = container_nan_n_inf.has_nans(leafwise=True, include_infs=False)
    assert container_nan_n_inf_hn.a is False
    assert container_nan_n_inf_hn.b.c is True
    assert container_nan_n_inf_hn.b.d is False


def test_container_at_keys(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    new_container = container.at_keys(['a', 'c'])
    assert np.allclose(ivy.to_numpy(new_container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([2]))
    assert 'd' not in new_container['b']
    new_container = container.at_keys('c')
    assert 'a' not in new_container
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([2]))
    assert 'd' not in new_container['b']
    new_container = container.at_keys(['b'])
    assert 'a' not in new_container
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container['b']['d']), np.array([3]))


def test_container_at_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)

    # explicit function call
    sub_container = container.at_key_chain('b')
    assert np.allclose(ivy.to_numpy(sub_container['c']), np.array([2]))
    sub_container = container.at_key_chain('b/c')
    assert np.allclose(ivy.to_numpy(sub_container), np.array([2]))

    # overridden built-in function call
    sub_container = container['b']
    assert np.allclose(ivy.to_numpy(sub_container['c']), np.array([2]))
    sub_container = container['b/c']
    assert np.allclose(ivy.to_numpy(sub_container), np.array([2]))


def test_container_at_key_chains(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    target_cont = Container({'a': True, 'b': {'c': True}})
    new_container = container.at_key_chains(target_cont)
    assert np.allclose(ivy.to_numpy(new_container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([2]))
    assert 'd' not in new_container['b']
    new_container = container.at_key_chains(['b/c', 'b/d'])
    assert 'a' not in new_container
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container['b']['d']), np.array([3]))
    new_container = container.at_key_chains('b/c')
    assert 'a' not in new_container
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([2]))
    assert 'd' not in new_container['b']


# noinspection PyUnresolvedReferences
def test_container_set_at_keys(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container_orig = Container(dict_in)

    # explicit function call
    orig_container = container_orig.copy()
    container = orig_container.set_at_keys({'b': ivy.array([4], dev_str=dev_str)})
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']), np.array([4]))
    assert not container.has_key('c')
    assert not container.has_key('d')
    container = orig_container.set_at_keys({'a': ivy.array([5], dev_str=dev_str), 'c': ivy.array([6], dev_str=dev_str)})
    assert np.allclose(ivy.to_numpy(container['a']), np.array([5]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([6]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))


# noinspection PyUnresolvedReferences
def test_container_set_at_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container_orig = Container(dict_in)

    # explicit function call
    container = container_orig.copy()
    container = container.set_at_key_chain('b/e', ivy.array([4], dev_str=dev_str))
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container['b']['e']), np.array([4]))
    container = container.set_at_key_chain('f', ivy.array([5], dev_str=dev_str))
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container['b']['e']), np.array([4]))
    assert np.allclose(ivy.to_numpy(container['f']), np.array([5]))

    # overridden built-in function call
    container = container_orig.copy()
    assert 'b/e' not in container
    container['b/e'] = ivy.array([4], dev_str=dev_str)
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container['b']['e']), np.array([4]))
    assert 'f' not in container
    container['f'] = ivy.array([5], dev_str=dev_str)
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container['b']['e']), np.array([4]))
    assert np.allclose(ivy.to_numpy(container['f']), np.array([5]))


# noinspection PyUnresolvedReferences
def test_container_overwrite_at_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container_orig = Container(dict_in)

    # explicit function call
    container = container_orig.copy()
    # noinspection PyBroadException
    try:
        container.overwrite_at_key_chain('b/e', ivy.array([4], dev_str=dev_str))
        exception_raised = False
    except Exception:
        exception_raised = True
    assert exception_raised
    container = container.overwrite_at_key_chain('b/d', ivy.array([4], dev_str=dev_str))
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([4]))


def test_container_set_at_key_chains(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    target_container = Container({'a': ivy.array([4], dev_str=dev_str),
                                  'b': {'d': ivy.array([5], dev_str=dev_str)}})
    new_container = container.set_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container['a']), np.array([4]))
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container['b']['d']), np.array([5]))
    target_container = Container({'b': {'c': ivy.array([7], dev_str=dev_str)}})
    new_container = container.set_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([7]))
    assert np.allclose(ivy.to_numpy(new_container['b']['d']), np.array([3]))


def test_container_overwrite_at_key_chains(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    target_container = Container({'a': ivy.array([4], dev_str=dev_str),
                                  'b': {'d': ivy.array([5], dev_str=dev_str)}})
    new_container = container.overwrite_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container['a']), np.array([4]))
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(new_container['b']['d']), np.array([5]))
    target_container = Container({'b': {'c': ivy.array([7], dev_str=dev_str)}})
    new_container = container.overwrite_at_key_chains(target_container, inplace=False)
    assert np.allclose(ivy.to_numpy(new_container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(new_container['b']['c']), np.array([7]))
    assert np.allclose(ivy.to_numpy(new_container['b']['d']), np.array([3]))
    # noinspection PyBroadException
    try:
        container.overwrite_at_key_chains(Container({'b': {'e': ivy.array([5], dev_str=dev_str)}}))
        exception_raised = False
    except Exception:
        exception_raised = True
    assert exception_raised


def test_container_prune_keys(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    container_pruned = container.prune_keys(['a', 'c'])
    assert 'a' not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned['b']['d']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
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

    def _test_bd_exception(container_in):
        try:
            _ = container_in.b.d
            return False
        except AttributeError:
            return True

    assert _test_a_exception(container_pruned)
    assert _test_bc_exception(container_pruned)

    container_pruned = container.prune_keys(['a', 'd'])
    assert 'a' not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned['b']['c']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.c), np.array([[2]]))
    assert 'd' not in container_pruned['b']
    assert _test_a_exception(container_pruned)
    assert _test_bd_exception(container_pruned)


def test_container_prune_key_chain(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': None}}
    container = Container(dict_in)
    container_pruned = container.prune_key_chain('b/c')
    assert np.allclose(ivy.to_numpy(container_pruned['a']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.a), np.array([[1]]))
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
    assert np.allclose(ivy.to_numpy(container_pruned['a']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.a), np.array([[1]]))
    assert ('b' not in container_pruned.keys())

    def _test_exception(container_in):
        try:
            _ = container_in.b
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)


def test_container_prune_key_chains(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    container_pruned = container.prune_key_chains(['a', 'b/c'])
    assert 'a' not in container_pruned
    assert np.allclose(ivy.to_numpy(container_pruned['b']['d']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
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
    assert np.allclose(ivy.to_numpy(container_pruned['b']['d']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
    assert 'c' not in container_pruned['b']
    assert _test_a_exception(container_pruned)
    assert _test_bc_exception(container_pruned)


def test_container_restructure_keys(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})

    # without base restructure
    container_restructured = container.restructure_keys([('a', 'a/new'), ('b/c', 'B/C'), ('b/d', 'Bee/Dee')])
    assert np.allclose(ivy.to_numpy(container_restructured['a']['new']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_restructured.a.new), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_restructured['B']['C']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_restructured.B.C), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_restructured['Bee']['Dee']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_restructured.Bee.Dee), np.array([3]))

    # with base restructure
    container_restructured = container.restructure_keys([('', 'new_base')])
    assert np.allclose(ivy.to_numpy(container_restructured['new_base']['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_restructured.new_base.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container_restructured['new_base']['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_restructured.new_base.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container_restructured['new_base']['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container_restructured.new_base.b.d), np.array([3]))


def test_container_prune_empty(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': {}, 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    container_pruned = container.prune_empty()
    assert np.allclose(ivy.to_numpy(container_pruned['a']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned['b']['d']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.b.d), np.array([[3]]))
    assert ('c' not in container_pruned['b'])

    def _test_exception(container_in):
        try:
            _ = container_in.b.c
            return False
        except AttributeError:
            return True

    assert _test_exception(container_pruned)


def test_container_prune_key_from_key_chains(dev_str, call):
    container = Container({'Ayy': ivy.array([1], dev_str=dev_str),
                           'Bee': {'Cee': ivy.array([2], dev_str=dev_str), 'Dee': ivy.array([3], dev_str=dev_str)},
                           'Beh': {'Ceh': ivy.array([4], dev_str=dev_str), 'Deh': ivy.array([5], dev_str=dev_str)}})

    # absolute
    container_pruned = container.prune_key_from_key_chains('Bee')
    assert np.allclose(ivy.to_numpy(container_pruned['Ayy']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Cee']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Dee']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert ('Bee' not in container_pruned)

    # containing
    container_pruned = container.prune_key_from_key_chains(containing='B')
    assert np.allclose(ivy.to_numpy(container_pruned['Ayy']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Cee']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Dee']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Ceh']), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ceh), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Deh']), np.array([[5]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Deh), np.array([[5]]))
    assert ('Bee' not in container_pruned)
    assert ('Beh' not in container_pruned)


def test_container_prune_keys_from_key_chains(dev_str, call):
    container = Container({'Ayy': ivy.array([1], dev_str=dev_str),
                           'Bee': {'Cee': ivy.array([2], dev_str=dev_str), 'Dee': ivy.array([3], dev_str=dev_str)},
                           'Eee': {'Fff': ivy.array([4], dev_str=dev_str)}})

    # absolute
    container_pruned = container.prune_keys_from_key_chains(['Bee', 'Eee'])
    assert np.allclose(ivy.to_numpy(container_pruned['Ayy']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Cee']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Dee']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Fff']), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Fff), np.array([[4]]))
    assert ('Bee' not in container_pruned)
    assert ('Eee' not in container_pruned)

    # containing
    container_pruned = container.prune_keys_from_key_chains(containing=['B', 'E'])
    assert np.allclose(ivy.to_numpy(container_pruned['Ayy']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Ayy), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Cee']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Cee), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Dee']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Dee), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_pruned['Fff']), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_pruned.Fff), np.array([[4]]))
    assert ('Bee' not in container_pruned)
    assert ('Eee' not in container_pruned)


def test_container_contains(dev_str, call):
    dict_in = {'a': ivy.array([0.], dev_str=dev_str),
               'b': {'c': ivy.array([1.], dev_str=dev_str), 'd': ivy.array([2.], dev_str=dev_str)}}
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
    dict_in = {'a': ivy.array([1, 2, 3], dev_str=dev_str),
               'b': {'c': ivy.array([1, 2, 3], dev_str=dev_str), 'd': ivy.array([1, 2, 3], dev_str=dev_str)}}
    container = Container(dict_in)

    # without key_chains specification
    container_shuffled = container.shuffle(0)
    data = ivy.array([1, 2, 3], dev_str=dev_str)
    ivy.core.random.seed()
    shuffled_data = ivy.to_numpy(ivy.core.random.shuffle(data))
    assert (ivy.to_numpy(container_shuffled['a']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.a) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled['b']['c']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.b.c) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled['b']['d']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.b.d) == shuffled_data).all()

    # with key_chains to apply
    container_shuffled = container.shuffle(0, ['a', 'b/c'])
    data = ivy.array([1, 2, 3], dev_str=dev_str)
    ivy.core.random.seed()
    shuffled_data = ivy.to_numpy(ivy.core.random.shuffle(data))
    assert (ivy.to_numpy(container_shuffled['a']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.a) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled['b']['c']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.b.c) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled['b']['d']) == ivy.to_numpy(data)).all()
    assert (ivy.to_numpy(container_shuffled.b.d) == ivy.to_numpy(data)).all()

    # with key_chains to apply pruned
    container_shuffled = container.shuffle(0, ['a', 'b/c'], prune_unapplied=True)
    data = ivy.array([1, 2, 3], dev_str=dev_str)
    ivy.core.random.seed()
    shuffled_data = ivy.to_numpy(ivy.core.random.shuffle(data))
    assert (ivy.to_numpy(container_shuffled['a']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.a) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled['b']['c']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.b.c) == shuffled_data).all()
    assert 'b/d' not in container_shuffled

    # with key_chains to not apply pruned
    container_shuffled = container.shuffle(0, Container({'a': None, 'b': {'d': None}}), to_apply=False)
    data = ivy.array([1, 2, 3], dev_str=dev_str)
    ivy.core.random.seed()
    shuffled_data = ivy.to_numpy(ivy.core.random.shuffle(data))
    assert (ivy.to_numpy(container_shuffled['a']) == ivy.to_numpy(data)).all()
    assert (ivy.to_numpy(container_shuffled.a) == ivy.to_numpy(data)).all()
    assert (ivy.to_numpy(container_shuffled['b']['c']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.b.c) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled['b']['d']) == ivy.to_numpy(data)).all()
    assert (ivy.to_numpy(container_shuffled.b.d) == ivy.to_numpy(data)).all()

    # with key_chains to not apply pruned
    container_shuffled = container.shuffle(0, Container({'a': None, 'b': {'d': None}}), to_apply=False,
                                           prune_unapplied=True)
    data = ivy.array([1, 2, 3], dev_str=dev_str)
    ivy.core.random.seed()
    shuffled_data = ivy.to_numpy(ivy.core.random.shuffle(data))
    assert 'a' not in container_shuffled
    assert (ivy.to_numpy(container_shuffled['b']['c']) == shuffled_data).all()
    assert (ivy.to_numpy(container_shuffled.b.c) == shuffled_data).all()
    assert 'b/d' not in container_shuffled


def test_container_to_iterator(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)

    # with key chains
    container_iterator = container.to_iterator()
    for (key_chain, value), expected in zip(
            container_iterator, [('a', ivy.array([1], dev_str=dev_str)), ('b/c', ivy.array([2], dev_str=dev_str)),
                                 ('b/d', ivy.array([3], dev_str=dev_str))]):
        expected_key_chain = expected[0]
        expected_value = expected[1]
        assert key_chain == expected_key_chain
        assert value == expected_value

    # with leaf keys
    container_iterator = container.to_iterator(leaf_keys_only=True)
    for (key_chain, value), expected in zip(
            container_iterator, [('a', ivy.array([1], dev_str=dev_str)), ('c', ivy.array([2], dev_str=dev_str)),
                                 ('d', ivy.array([3], dev_str=dev_str))]):
        expected_key_chain = expected[0]
        expected_value = expected[1]
        assert key_chain == expected_key_chain
        assert value == expected_value


def test_container_to_flat_list(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    container_flat_list = container.to_flat_list()
    for value, expected_value in zip(container_flat_list,
                                     [ivy.array([1], dev_str=dev_str), ivy.array([2], dev_str=dev_str),
                                      ivy.array([3], dev_str=dev_str)]):
        assert value == expected_value


def test_container_from_flat_list(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    flat_list = [4, 5, 6]
    container = container.from_flat_list(flat_list)
    assert np.allclose(ivy.to_numpy(container['a']), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([4]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([6]))


def test_container_map(dev_str, call):
    # without key_chains specification
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    container_iterator = container.map(lambda x, _: x + 1).to_iterator()
    for (key, value), expected_value in zip(container_iterator,
                                            [ivy.array([2], dev_str=dev_str), ivy.array([3], dev_str=dev_str),
                                             ivy.array([4], dev_str=dev_str)]):
        assert call(lambda x: x, value) == call(lambda x: x, expected_value)

    # with key_chains to apply
    container_mapped = container.map(lambda x, _: x + 1, ['a', 'b/c'])
    assert np.allclose(ivy.to_numpy(container_mapped['a']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped['b']['c']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped['b']['d']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.d), np.array([[3]]))

    # with key_chains to apply pruned
    container_mapped = container.map(lambda x, _: x + 1, ['a', 'b/c'], prune_unapplied=True)
    assert np.allclose(ivy.to_numpy(container_mapped['a']), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[2]]))
    assert np.allclose(ivy.to_numpy(container_mapped['b']['c']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    assert 'b/d' not in container_mapped

    # with key_chains to not apply
    container_mapped = container.map(lambda x, _: x + 1, Container({'a': None, 'b': {'d': None}}), to_apply=False)
    assert np.allclose(ivy.to_numpy(container_mapped['a']), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[1]]))
    assert np.allclose(ivy.to_numpy(container_mapped['b']['c']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped['b']['d']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.d), np.array([[3]]))

    # with key_chains to not apply pruned
    container_mapped = container.map(lambda x, _: x + 1, Container({'a': None, 'b': {'d': None}}), to_apply=False,
                                     prune_unapplied=True)
    assert 'a' not in container_mapped
    assert np.allclose(ivy.to_numpy(container_mapped['b']['c']), np.array([[3]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[3]]))
    assert 'b/d' not in container_mapped


def test_container_map_conts(dev_str, call):
    # without key_chains specification
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})

    def _add_e_attr(cont_in):
        cont_in.e = ivy.array([4], dev_str=dev_str)
        return cont_in

    # with self
    container_mapped = container.map_conts(lambda c, _: _add_e_attr(c))
    assert 'e' in container_mapped
    assert np.array_equal(ivy.to_numpy(container_mapped.e), np.array([4]))
    assert 'e' in container_mapped.b
    assert np.array_equal(ivy.to_numpy(container_mapped.b.e), np.array([4]))

    # without self
    container_mapped = container.map_conts(lambda c, _: _add_e_attr(c), include_self=False)
    assert 'e' not in container_mapped
    assert 'e' in container_mapped.b
    assert np.array_equal(ivy.to_numpy(container_mapped.b.e), np.array([4]))


def test_container_multi_map(dev_str, call):
    # without key_chains specification
    container0 = Container({'a': ivy.array([1], dev_str=dev_str),
                            'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container1 = Container({'a': ivy.array([3], dev_str=dev_str),
                            'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})

    # with key_chains to apply
    container_mapped = ivy.Container.multi_map(lambda x, _: x[0] + x[1], [container0, container1])
    assert np.allclose(ivy.to_numpy(container_mapped['a']), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_mapped.a), np.array([[4]]))
    assert np.allclose(ivy.to_numpy(container_mapped['b']['c']), np.array([[6]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.c), np.array([[6]]))
    assert np.allclose(ivy.to_numpy(container_mapped['b']['d']), np.array([[8]]))
    assert np.allclose(ivy.to_numpy(container_mapped.b.d), np.array([[8]]))


def test_container_identical_structure(dev_str, call):
    # without key_chains specification
    container0 = Container({'a': ivy.array([1], dev_str=dev_str),
                            'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container1 = Container({'a': ivy.array([3], dev_str=dev_str),
                            'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container2 = Container({'a': ivy.array([3], dev_str=dev_str),
                            'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str),
                                  'e': ivy.array([6], dev_str=dev_str)}})
    container3 = Container({'a': ivy.array([3], dev_str=dev_str),
                            'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)},
                            'e': ivy.array([6], dev_str=dev_str)})

    # with identical
    assert ivy.Container.identical_structure([container0, container1])
    assert ivy.Container.identical_structure([container1, container0])
    assert ivy.Container.identical_structure([container1, container0, container1])

    # without identical
    assert not ivy.Container.identical_structure([container2, container3])
    assert not ivy.Container.identical_structure([container0, container3])
    assert not ivy.Container.identical_structure([container1, container2])
    assert not ivy.Container.identical_structure([container1, container0, container2])


def test_container_dtype(dev_str, call):
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2.], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}
    container = Container(dict_in)
    dtype_container = container.dtype()
    for (key, value), expected_value in zip(dtype_container.to_iterator(),
                                            [ivy.array([1], dev_str=dev_str).dtype,
                                             ivy.array([2.], dev_str=dev_str).dtype,
                                             ivy.array([3], dev_str=dev_str).dtype]):
        assert value == expected_value


def test_container_with_entries_as_lists(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # to_list() requires eager execution
        pytest.skip()
    dict_in = {'a': ivy.array([1], dev_str=dev_str),
               'b': {'c': ivy.array([2.], dev_str=dev_str), 'd': 'some string'}}
    container = Container(dict_in)
    container_w_list_entries = container.with_entries_as_lists()
    for (key, value), expected_value in zip(container_w_list_entries.to_iterator(),
                                            [[1],
                                             [2.],
                                             'some string']):
        assert value == expected_value


def test_container_reshape_like(dev_str, call):
    container = Container({'a': ivy.array([[1.]], dev_str=dev_str),
                           'b': {'c': ivy.array([[3.], [4.]], dev_str=dev_str),
                                 'd': ivy.array([[5.], [6.], [7.]], dev_str=dev_str)}})
    new_shapes = Container({'a': (1,),
                            'b': {'c': (1, 2, 1), 'd': (3, 1, 1)}})

    # without leading shape
    container_reshaped = container.reshape_like(new_shapes)
    assert list(container_reshaped['a'].shape) == [1]
    assert list(container_reshaped.a.shape) == [1]
    assert list(container_reshaped['b']['c'].shape) == [1, 2, 1]
    assert list(container_reshaped.b.c.shape) == [1, 2, 1]
    assert list(container_reshaped['b']['d'].shape) == [3, 1, 1]
    assert list(container_reshaped.b.d.shape) == [3, 1, 1]

    # with leading shape
    container = Container({'a': ivy.array([[[1.]], [[1.]], [[1.]]], dev_str=dev_str),
                           'b': {'c': ivy.array([[[3.], [4.]], [[3.], [4.]], [[3.], [4.]]], dev_str=dev_str),
                                 'd': ivy.array([[[5.], [6.], [7.]], [[5.], [6.], [7.]], [[5.], [6.], [7.]]],
                                                dev_str=dev_str)}})
    container_reshaped = container.reshape_like(new_shapes, leading_shape=[3])
    assert list(container_reshaped['a'].shape) == [3, 1]
    assert list(container_reshaped.a.shape) == [3, 1]
    assert list(container_reshaped['b']['c'].shape) == [3, 1, 2, 1]
    assert list(container_reshaped.b.c.shape) == [3, 1, 2, 1]
    assert list(container_reshaped['b']['d'].shape) == [3, 3, 1, 1]
    assert list(container_reshaped.b.d.shape) == [3, 3, 1, 1]


def test_container_slice(dev_str, call):
    dict_in = {'a': ivy.array([[0.], [1.]], dev_str=dev_str),
               'b': {'c': ivy.array([[1.], [2.]], dev_str=dev_str), 'd': ivy.array([[2.], [3.]], dev_str=dev_str)}}
    container = Container(dict_in)
    container0 = container[0]
    container1 = container[1]
    assert np.array_equal(ivy.to_numpy(container0['a']), np.array([0.]))
    assert np.array_equal(ivy.to_numpy(container0.a), np.array([0.]))
    assert np.array_equal(ivy.to_numpy(container0['b']['c']), np.array([1.]))
    assert np.array_equal(ivy.to_numpy(container0.b.c), np.array([1.]))
    assert np.array_equal(ivy.to_numpy(container0['b']['d']), np.array([2.]))
    assert np.array_equal(ivy.to_numpy(container0.b.d), np.array([2.]))
    assert np.array_equal(ivy.to_numpy(container1['a']), np.array([1.]))
    assert np.array_equal(ivy.to_numpy(container1.a), np.array([1.]))
    assert np.array_equal(ivy.to_numpy(container1['b']['c']), np.array([2.]))
    assert np.array_equal(ivy.to_numpy(container1.b.c), np.array([2.]))
    assert np.array_equal(ivy.to_numpy(container1['b']['d']), np.array([3.]))
    assert np.array_equal(ivy.to_numpy(container1.b.d), np.array([3.]))


def test_container_slice_via_key(dev_str, call):
    dict_in = {'a': {'x': ivy.array([0.], dev_str=dev_str),
                     'y': ivy.array([1.], dev_str=dev_str)},
               'b': {'c': {'x': ivy.array([1.], dev_str=dev_str),
                           'y': ivy.array([2.], dev_str=dev_str)},
                     'd': {'x': ivy.array([2.], dev_str=dev_str),
                           'y': ivy.array([3.], dev_str=dev_str)}}}
    container = Container(dict_in)
    containerx = container.slice_via_key('x')
    containery = container.slice_via_key('y')
    assert np.array_equal(ivy.to_numpy(containerx['a']), np.array([0.]))
    assert np.array_equal(ivy.to_numpy(containerx.a), np.array([0.]))
    assert np.array_equal(ivy.to_numpy(containerx['b']['c']), np.array([1.]))
    assert np.array_equal(ivy.to_numpy(containerx.b.c), np.array([1.]))
    assert np.array_equal(ivy.to_numpy(containerx['b']['d']), np.array([2.]))
    assert np.array_equal(ivy.to_numpy(containerx.b.d), np.array([2.]))
    assert np.array_equal(ivy.to_numpy(containery['a']), np.array([1.]))
    assert np.array_equal(ivy.to_numpy(containery.a), np.array([1.]))
    assert np.array_equal(ivy.to_numpy(containery['b']['c']), np.array([2.]))
    assert np.array_equal(ivy.to_numpy(containery.b.c), np.array([2.]))
    assert np.array_equal(ivy.to_numpy(containery['b']['d']), np.array([3.]))
    assert np.array_equal(ivy.to_numpy(containery.b.d), np.array([3.]))


def test_container_to_and_from_disk_as_hdf5(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.hdf5'
    dict_in_1 = {'a': ivy.array([np.float32(1.)], dev_str=dev_str),
                 'b': {'c': ivy.array([np.float32(2.)], dev_str=dev_str),
                       'd': ivy.array([np.float32(3.)], dev_str=dev_str)}}
    container1 = Container(dict_in_1)
    dict_in_2 = {'a': ivy.array([np.float32(1.), np.float32(1.)], dev_str=dev_str),
                 'b': {'c': ivy.array([np.float32(2.), np.float32(2.)], dev_str=dev_str),
                       'd': ivy.array([np.float32(3.), np.float32(3.)], dev_str=dev_str)}}
    container2 = Container(dict_in_2)

    # saving
    container1.to_disk_as_hdf5(save_filepath, max_batch_size=2)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_hdf5(save_filepath, slice(1))
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container1.a))
    assert np.array_equal(ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container1.b.c))
    assert np.array_equal(ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container1.b.d))

    # appending
    container1.to_disk_as_hdf5(save_filepath, max_batch_size=2, starting_index=1)
    assert os.path.exists(save_filepath)

    # loading after append
    loaded_container = Container.from_disk_as_hdf5(save_filepath)
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container2.a))
    assert np.array_equal(ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container2.b.c))
    assert np.array_equal(ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container2.b.d))

    # load slice
    loaded_sliced_container = Container.from_disk_as_hdf5(save_filepath, slice(1, 2))
    assert np.array_equal(ivy.to_numpy(loaded_sliced_container.a), ivy.to_numpy(container1.a))
    assert np.array_equal(ivy.to_numpy(loaded_sliced_container.b.c), ivy.to_numpy(container1.b.c))
    assert np.array_equal(ivy.to_numpy(loaded_sliced_container.b.d), ivy.to_numpy(container1.b.d))

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
    dict_in = {'a': ivy.array([1, 2, 3], dev_str=dev_str),
               'b': {'c': ivy.array([1, 2, 3], dev_str=dev_str), 'd': ivy.array([1, 2, 3], dev_str=dev_str)}}
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
    dict_in = {'a': ivy.array([np.float32(1.)], dev_str=dev_str),
               'b': {'c': ivy.array([np.float32(2.)], dev_str=dev_str),
                     'd': ivy.array([np.float32(3.)], dev_str=dev_str)}}
    container = Container(dict_in)

    # saving
    container.to_disk_as_pickled(save_filepath)
    assert os.path.exists(save_filepath)

    # loading
    loaded_container = Container.from_disk_as_pickled(save_filepath)
    assert np.array_equal(ivy.to_numpy(loaded_container.a), ivy.to_numpy(container.a))
    assert np.array_equal(ivy.to_numpy(loaded_container.b.c), ivy.to_numpy(container.b.c))
    assert np.array_equal(ivy.to_numpy(loaded_container.b.d), ivy.to_numpy(container.b.d))

    os.remove(save_filepath)


def test_container_to_and_from_disk_as_json(dev_str, call):
    if call in [helpers.tf_graph_call]:
        # container disk saving requires eager execution
        pytest.skip()
    save_filepath = 'container_on_disk.json'
    dict_in = {'a': 1.274e-7, 'b': {'c': True, 'd': ivy.array([np.float32(3.)], dev_str=dev_str)}}
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
    container = +Container({'a': ivy.array([1], dev_str=dev_str),
                            'b': {'c': ivy.array([-2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([-2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([-2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_negative(dev_str, call):
    container = -Container({'a': ivy.array([1], dev_str=dev_str),
                            'b': {'c': ivy.array([-2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    assert np.allclose(ivy.to_numpy(container['a']), np.array([-1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([-1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([-3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([-3]))


def test_container_pow(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container = container_a ** container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([16]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([16]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([729]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([729]))


def test_container_scalar_pow(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = container_a ** 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([4]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([9]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([9]))


def test_container_reverse_scalar_pow(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 2 ** container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([4]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([8]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([8]))


def test_container_scalar_addition(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container += 3
    assert np.allclose(ivy.to_numpy(container['a']), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([4]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([6]))


def test_container_reverse_scalar_addition(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 3 + container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([4]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([4]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([6]))


def test_container_addition(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container = container_a + container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([3]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([6]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([9]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([9]))


def test_container_scalar_subtraction(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container -= 1
    assert np.allclose(ivy.to_numpy(container['a']), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([2]))


def test_container_reverse_scalar_subtraction(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 1 - container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([-1]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([-1]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([-2]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([-2]))


def test_container_subtraction(dev_str, call):
    container_a = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([1], dev_str=dev_str), 'd': ivy.array([4], dev_str=dev_str)}})
    container = container_a - container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([3]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([2]))


def test_container_sum(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container = sum([container_a, container_b])
    assert np.allclose(ivy.to_numpy(container['a']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([3]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([6]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([6]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([9]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([9]))


def test_container_scalar_multiplication(dev_str, call):
    container = Container({'a': ivy.array([1.], dev_str=dev_str),
                           'b': {'c': ivy.array([2.], dev_str=dev_str), 'd': ivy.array([3.], dev_str=dev_str)}})
    container *= 2.5
    assert np.allclose(ivy.to_numpy(container['a']), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([5.]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5.]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([7.5]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([7.5]))


def test_container_reverse_scalar_multiplication(dev_str, call):
    container = Container({'a': ivy.array([1.], dev_str=dev_str),
                           'b': {'c': ivy.array([2.], dev_str=dev_str), 'd': ivy.array([3.], dev_str=dev_str)}})
    container = 2.5 * container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([5.]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5.]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([7.5]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([7.5]))


def test_container_multiplication(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([4], dev_str=dev_str), 'd': ivy.array([6], dev_str=dev_str)}})
    container = container_a * container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([8]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([8]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([18]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([18]))


def test_container_scalar_truediv(dev_str, call):
    container = Container({'a': ivy.array([1.], dev_str=dev_str),
                           'b': {'c': ivy.array([5.], dev_str=dev_str), 'd': ivy.array([5.], dev_str=dev_str)}})
    container /= 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([0.5]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0.5]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([2.5]))


def test_container_reverse_scalar_truediv(dev_str, call):
    container = Container({'a': ivy.array([1.], dev_str=dev_str),
                           'b': {'c': ivy.array([5.], dev_str=dev_str), 'd': ivy.array([5.], dev_str=dev_str)}})
    container = 2 / container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([2.]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2.]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([0.4]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([0.4]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([0.4]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([0.4]))


def test_container_truediv(dev_str, call):
    container_a = Container({'a': ivy.array([1.], dev_str=dev_str),
                             'b': {'c': ivy.array([5.], dev_str=dev_str), 'd': ivy.array([5.], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2.], dev_str=dev_str),
                             'b': {'c': ivy.array([2.], dev_str=dev_str), 'd': ivy.array([4.], dev_str=dev_str)}})
    container = container_a / container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([0.5]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0.5]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2.5]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([1.25]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([1.25]))


def test_container_scalar_floordiv(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit ivy.floordiv is implemented at some point
        pytest.skip()
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container //= 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([2]))


def test_container_reverse_scalar_floordiv(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit ivy.floordiv is implemented at some point
        pytest.skip()
    container = Container({'a': ivy.array([2], dev_str=dev_str),
                           'b': {'c': ivy.array([1], dev_str=dev_str), 'd': ivy.array([7], dev_str=dev_str)}})
    container = 5 // container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([5]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([5]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([0]))


def test_container_floordiv(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the // operator, can add if explicit ivy.floordiv is implemented at some point
        pytest.skip()
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([4], dev_str=dev_str)}})
    container = container_a // container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([0]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([0]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([1]))


def test_container_abs(dev_str, call):
    container = abs(Container({'a': ivy.array([1], dev_str=dev_str),
                               'b': {'c': ivy.array([-2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}}))
    assert np.allclose(ivy.to_numpy(container['a']), np.array([1]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([1]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([2]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([2]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([3]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([3]))


def test_container_scalar_less_than(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = container < 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_less_than(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 2 < container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_less_than(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container = container_a < container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_scalar_less_than_or_equal_to(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = container <= 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_less_than_or_equal_to(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 2 <= container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_less_than_or_equal_to(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container = container_a <= container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_scalar_equal_to(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = container == 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_equal_to(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 2 == container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_equal_to(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container = container_a == container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_scalar_not_equal_to(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = container != 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_reverse_scalar_not_equal_to(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 2 != container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_not_equal_to(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container = container_a != container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_scalar_greater_than(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = container > 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_reverse_scalar_greater_than(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 2 > container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_greater_than(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container = container_a > container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_scalar_greater_than_or_equal_to(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = container >= 2
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_reverse_scalar_greater_than_or_equal_to(dev_str, call):
    container = Container({'a': ivy.array([1], dev_str=dev_str),
                           'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([3], dev_str=dev_str)}})
    container = 2 >= container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_greater_than_or_equal_to(dev_str, call):
    container_a = Container({'a': ivy.array([1], dev_str=dev_str),
                             'b': {'c': ivy.array([5], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([2], dev_str=dev_str),
                             'b': {'c': ivy.array([2], dev_str=dev_str), 'd': ivy.array([5], dev_str=dev_str)}})
    container = container_a >= container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_scalar_and(dev_str, call):
    container = Container({'a': ivy.array([True], dev_str=dev_str),
                           'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = container & True
    # ToDo: work out why "container and True" does not work. Perhaps bool(container) is called first implicitly?
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_and(dev_str, call):
    container = Container({'a': ivy.array([True], dev_str=dev_str),
                           'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = True and container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_and(dev_str, call):
    container_a = Container({'a': ivy.array([True], dev_str=dev_str),
                             'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([False], dev_str=dev_str),
                             'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = container_a and container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_scalar_or(dev_str, call):
    container = Container({'a': ivy.array([True], dev_str=dev_str),
                           'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = container or False
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_reverse_scalar_or(dev_str, call):
    container = Container({'a': ivy.array([True], dev_str=dev_str),
                           'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = container or False
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_or(dev_str, call):
    container_a = Container({'a': ivy.array([True], dev_str=dev_str),
                             'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([False], dev_str=dev_str),
                             'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = container_a or container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_not(dev_str, call):
    container = ~Container({'a': ivy.array([True], dev_str=dev_str),
                            'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_scalar_xor(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the ^ operator, can add if explicit ivy.logical_xor is implemented at some point
        pytest.skip()
    container = Container({'a': ivy.array([True], dev_str=dev_str),
                           'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = container != True
    assert np.allclose(ivy.to_numpy(container['a']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([True]))


def test_container_reverse_scalar_xor(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the ^ operator, can add if explicit ivy.logical_xor is implemented at some point
        pytest.skip()
    container = Container({'a': ivy.array([True], dev_str=dev_str),
                           'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = False != container
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_xor(dev_str, call):
    if call is helpers.mx_call:
        # MXnet arrays do not overload the ^ operator, can add if explicit ivy.logical_xor is implemented at some point
        pytest.skip()
    container_a = Container({'a': ivy.array([True], dev_str=dev_str),
                             'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container_b = Container({'a': ivy.array([False], dev_str=dev_str),
                             'b': {'c': ivy.array([True], dev_str=dev_str), 'd': ivy.array([False], dev_str=dev_str)}})
    container = container_a != container_b
    assert np.allclose(ivy.to_numpy(container['a']), np.array([True]))
    assert np.allclose(ivy.to_numpy(container.a), np.array([True]))
    assert np.allclose(ivy.to_numpy(container['b']['c']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.c), np.array([False]))
    assert np.allclose(ivy.to_numpy(container['b']['d']), np.array([False]))
    assert np.allclose(ivy.to_numpy(container.b.d), np.array([False]))


def test_container_shape(dev_str, call):
    dict_in = {'a': ivy.array([[[1.], [2.], [3.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[2.], [4.], [6.]]], dev_str=dev_str),
                     'd': ivy.array([[[3.], [6.], [9.]]], dev_str=dev_str)}}
    container = Container(dict_in)
    assert container.shape == [1, 3, 1]
    dict_in = {'a': ivy.array([[[1.], [2.], [3.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[2., 3.], [4., 5.], [6., 7.]]], dev_str=dev_str),
                     'd': ivy.array([[[3.], [6.], [9.]]], dev_str=dev_str)}}
    container = Container(dict_in)
    assert container.shape == [1, 3, None]
    dict_in = {'a': ivy.array([[[1., 2.], [2., 3.], [3., 4.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[2., 3.], [4., 5.], [6., 7.]]], dev_str=dev_str),
                     'd': ivy.array([[[3., 4.], [6., 7.], [9., 10.]]], dev_str=dev_str)}}
    container = Container(dict_in)
    assert container.shape == [1, 3, 2]


def test_container_shapes(dev_str, call):
    dict_in = {'a': ivy.array([[[1.], [2.], [3.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[2.], [4.]]], dev_str=dev_str), 'd': ivy.array([[9.]], dev_str=dev_str)}}
    container_shapes = Container(dict_in).shapes
    assert list(container_shapes['a']) == [1, 3, 1]
    assert list(container_shapes.a) == [1, 3, 1]
    assert list(container_shapes['b']['c']) == [1, 2, 1]
    assert list(container_shapes.b.c) == [1, 2, 1]
    assert list(container_shapes['b']['d']) == [1, 1]
    assert list(container_shapes.b.d) == [1, 1]


def test_container_dev_str(dev_str, call):
    dict_in = {'a': ivy.array([[[1.], [2.], [3.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[2.], [4.], [6.]]], dev_str=dev_str),
                     'd': ivy.array([[[3.], [6.], [9.]]], dev_str=dev_str)}}
    container = Container(dict_in)
    assert container.dev_str == dev_str


def test_container_if_exists(dev_str, call):
    dict_in = {'a': ivy.array([[[1.], [2.], [3.]]], dev_str=dev_str),
               'b': {'c': ivy.array([[[2.], [4.], [6.]]], dev_str=dev_str),
                     'd': ivy.array([[[3.], [6.], [9.]]], dev_str=dev_str)}}
    container = Container(dict_in)
    assert np.allclose(ivy.to_numpy(container.if_exists('a')), np.array([[[1.], [2.], [3.]]]))
    assert 'c' not in container
    assert container.if_exists('c') is None
    container['c'] = ivy.array([[[1.], [2.], [3.]]], dev_str=dev_str)
    assert np.allclose(ivy.to_numpy(container.if_exists('c')), np.array([[[1.], [2.], [3.]]]))
    assert container.if_exists('d') is None
    container.d = ivy.array([[[1.], [2.], [3.]]], dev_str=dev_str)
    assert np.allclose(ivy.to_numpy(container.if_exists('d')), np.array([[[1.], [2.], [3.]]]))


def test_container_pickling_with_ivy_attribute(dev_str, call):
    # verify container with local ivy cannot be pickled
    cannot_pickle = False
    try:
        pickle.dumps(Container(ivyh=ivy.get_framework(ivy.current_framework_str())))
    except TypeError:
        cannot_pickle = True
    assert cannot_pickle

    # verify container without local ivy can be pickled
    pickle.dumps(Container())


def test_container_from_queues(dev_str, call):

    if 'gpu' in dev_str:
        # Cannot re-initialize CUDA in forked subprocess. 'spawn' start method must be used.
        pytest.skip()

    def worker_fn(in_queue, out_queue, load_size, worker_id):
        keep_going = True
        while keep_going:
            try:
                keep_going = in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            out_queue.put({'a': [ivy.to_native(ivy.array([1., 2., 3.], dev_str=dev_str)) * worker_id] * load_size})

    workers = list()
    in_queues = list()
    out_queues = list()
    queue_load_sizes = [1, 2, 1]
    for i, queue_load_size in enumerate(queue_load_sizes):
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()
        worker = multiprocessing.Process(target=worker_fn, args=(input_queue, output_queue, queue_load_size, i + 1))
        worker.start()
        in_queues.append(input_queue)
        out_queues.append(output_queue)
        workers.append(worker)

    container = Container(queues=out_queues, queue_load_sizes=queue_load_sizes, queue_timeout=0.25)

    # queue 0
    queue_was_empty = False
    try:
        container[0]
    except queue.Empty:
        queue_was_empty = True
    assert queue_was_empty
    in_queues[0].put(True)
    assert np.allclose(ivy.to_numpy(container[0].a), np.array([1., 2., 3.]))
    assert np.allclose(ivy.to_numpy(container[0].a), np.array([1., 2., 3.]))

    # queue 1
    queue_was_empty = False
    try:
        container[1]
    except queue.Empty:
        queue_was_empty = True
    assert queue_was_empty
    queue_was_empty = False
    try:
        container[2]
    except queue.Empty:
        queue_was_empty = True
    assert queue_was_empty
    in_queues[1].put(True)
    assert np.allclose(ivy.to_numpy(container[1].a), np.array([2., 4., 6.]))
    assert np.allclose(ivy.to_numpy(container[1].a), np.array([2., 4., 6.]))
    assert np.allclose(ivy.to_numpy(container[2].a), np.array([2., 4., 6.]))
    assert np.allclose(ivy.to_numpy(container[2].a), np.array([2., 4., 6.]))

    # queue 2
    queue_was_empty = False
    try:
        container[3]
    except queue.Empty:
        queue_was_empty = True
    assert queue_was_empty
    in_queues[2].put(True)
    assert np.allclose(ivy.to_numpy(container[3].a), np.array([3., 6., 9.]))
    assert np.allclose(ivy.to_numpy(container[3].a), np.array([3., 6., 9.]))

    # stop workers
    in_queues[0].put(False)
    in_queues[1].put(False)
    in_queues[2].put(False)
    in_queues[0].close()
    in_queues[1].close()
    in_queues[2].close()

    # join workers
    for worker in workers:
        worker.join()

    del container
