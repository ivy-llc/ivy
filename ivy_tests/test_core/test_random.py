"""
Collection of tests for templated reduction functions
"""

# global
import numpy as np

# local
import ivy.core.general as ivy_gen
import ivy.core.random as ivy_rand
import ivy_tests.helpers as helpers


def test_random_uniform():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_rand.random_uniform, 0, 1, (1,), f=lib).shape,
                              np.random.uniform(0, 1, (1,)).shape)
        assert np.array_equal(call(ivy_rand.random_uniform, 0, 1, (1, 2), f=lib).shape,
                              np.random.uniform(0, 1, (1, 2)).shape)


def test_randint():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_rand.randint, 0, 1, (1,), f=lib).shape,
                              np.random.randint(0, 1, (1,)).shape)
        assert np.array_equal(call(ivy_rand.randint, 0, 1, (1, 2), f=lib).shape,
                              np.random.randint(0, 1, (1, 2)).shape)


def test_random_seed():
    for lib, call in helpers.calls:
        call(ivy_rand.seed, 0, f=lib)


def test_random_shuffle():
    for lib, call in helpers.calls:
        call(ivy_rand.seed, 0, f=lib)
        first_shuffle = call(ivy_rand.shuffle, ivy_gen.array([1, 2, 3], f=lib), lib)
        call(ivy_rand.seed, 0, f=lib)
        second_shuffle = call(ivy_rand.shuffle, ivy_gen.array([1, 2, 3], f=lib), lib)
        assert np.array(first_shuffle == second_shuffle).all()
