"""
Collection of tests for templated reduction functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_random_uniform(dev_str, call):
    assert np.array_equal(call(ivy.random_uniform, 0, 1, (1,)).shape,
                          np.random.uniform(0, 1, (1,)).shape)
    assert np.array_equal(call(ivy.random_uniform, 0, 1, (1, 2)).shape,
                          np.random.uniform(0, 1, (1, 2)).shape)
    if call is helpers.torch_call:
        # pytorch scripting does not support Union or Numbers for type hinting
        return
    helpers.assert_compilable(ivy.random_uniform)


def test_randint(dev_str, call):
    assert np.array_equal(call(ivy.randint, 0, 1, (1,)).shape,
                          np.random.randint(0, 1, (1,)).shape)
    assert np.array_equal(call(ivy.randint, 0, 1, (1, 2)).shape,
                          np.random.randint(0, 1, (1, 2)).shape)
    helpers.assert_compilable(ivy.randint)


def test_random_seed(dev_str, call):
    call(ivy.seed, 0)
    if call in [helpers.torch_call]:
        # pytorch scripting does not support functions with None return
        return
    helpers.assert_compilable(ivy.seed)


def test_random_shuffle(dev_str, call):
    call(ivy.seed, 0)
    first_shuffle = call(ivy.shuffle, ivy.tensor([1, 2, 3]))
    call(ivy.seed, 0)
    second_shuffle = call(ivy.shuffle, ivy.tensor([1, 2, 3]))
    assert np.array(first_shuffle == second_shuffle).all()
    helpers.assert_compilable(ivy.shuffle)
