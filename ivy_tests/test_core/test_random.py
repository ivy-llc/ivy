"""
Collection of tests for templated reduction functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_random_uniform():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.random_uniform, 0, 1, (1,), f=f).shape,
                              np.random.uniform(0, 1, (1,)).shape)
        assert np.array_equal(call(ivy.random_uniform, 0, 1, (1, 2), f=f).shape,
                              np.random.uniform(0, 1, (1, 2)).shape)
        if call is helpers.torch_call:
            # pytorch scripting does not support Union or Numbers for type hinting
            continue
        helpers.assert_compilable('random_uniform', f)


def test_randint():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.randint, 0, 1, (1,), f=f).shape,
                              np.random.randint(0, 1, (1,)).shape)
        assert np.array_equal(call(ivy.randint, 0, 1, (1, 2), f=f).shape,
                              np.random.randint(0, 1, (1, 2)).shape)
        helpers.assert_compilable('randint', f)


def test_random_seed():
    for f, call in helpers.f_n_calls():
        call(ivy.seed, 0, f=f)
        if call in [helpers.torch_call]:
            # pytorch scripting does not support functions with None return
            continue
        helpers.assert_compilable('seed', f)


def test_random_shuffle():
    for f, call in helpers.f_n_calls():
        call(ivy.seed, 0, f=f)
        first_shuffle = call(ivy.shuffle, ivy.array([1, 2, 3], f=f), f)
        call(ivy.seed, 0, f=f)
        second_shuffle = call(ivy.shuffle, ivy.array([1, 2, 3], f=f), f)
        assert np.array(first_shuffle == second_shuffle).all()
        helpers.assert_compilable('shuffle', f)
