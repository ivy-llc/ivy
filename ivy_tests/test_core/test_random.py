"""
Collection of tests for templated reduction functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_random_uniform():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.random_uniform, 0, 1, (1,), f=lib).shape,
                              np.random.uniform(0, 1, (1,)).shape)
        assert np.array_equal(call(ivy.random_uniform, 0, 1, (1, 2), f=lib).shape,
                              np.random.uniform(0, 1, (1, 2)).shape)
        if call is helpers.torch_call:
            # pytorch scripting does not support Union or Numbers for type hinting
            continue
        helpers.assert_compilable('random_uniform', lib)


def test_randint():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.randint, 0, 1, (1,), f=lib).shape,
                              np.random.randint(0, 1, (1,)).shape)
        assert np.array_equal(call(ivy.randint, 0, 1, (1, 2), f=lib).shape,
                              np.random.randint(0, 1, (1, 2)).shape)
        helpers.assert_compilable('randint', lib)


def test_random_seed():
    for lib, call in helpers.calls():
        call(ivy.seed, 0, f=lib)
        if call in [helpers.torch_call]:
            # pytorch scripting does not support functions with None return
            continue
        helpers.assert_compilable('seed', lib)


def test_random_shuffle():
    for lib, call in helpers.calls():
        call(ivy.seed, 0, f=lib)
        first_shuffle = call(ivy.shuffle, ivy.array([1, 2, 3], f=lib), lib)
        call(ivy.seed, 0, f=lib)
        second_shuffle = call(ivy.shuffle, ivy.array([1, 2, 3], f=lib), lib)
        assert np.array(first_shuffle == second_shuffle).all()
        helpers.assert_compilable('shuffle', lib)
