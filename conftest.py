import pytest
import ivy.jax
import ivy.mxnd
import ivy.torch
import ivy.numpy
import ivy.tensorflow
from typing import Dict
from types import ModuleType
from ivy_tests import helpers


FW_STRS = ['numpy', 'jax', 'tensorflow', 'tensorflow_graph', 'torch', 'mxnd']


TEST_FRAMEWORKS: Dict[str, ModuleType] = {'numpy': ivy.numpy,
                                          'jax': ivy.jax,
                                          'tensorflow': ivy.tensorflow,
                                          'tensorflow_graph': ivy.tensorflow,
                                          'torch': ivy.torch,
                                          'mxnd': ivy.mxnd}
TEST_CALL_METHODS: Dict[str, callable] = {'numpy': helpers.np_call,
                                          'jax': helpers.jnp_call,
                                          'tensorflow': helpers.tf_call,
                                          'tensorflow_graph': helpers.tf_graph_call,
                                          'torch': helpers.torch_call,
                                          'mxnd': helpers.mx_call}


@pytest.fixture(autouse=True)
def run_around_tests(f, wrapped_mode, call):
    if wrapped_mode and call is helpers.tf_graph_call:
        # ToDo: add support for wrapped_mode and tensorflow compilation
        pytest.skip()
    with f.use:
        f.set_wrapped_mode(wrapped_mode)
        yield


def pytest_generate_tests(metafunc):

    # dev_str
    raw_value = metafunc.config.getoption('--dev_str')
    if raw_value == 'all':
        dev_strs = ['cpu:0', 'gpu:0', 'tpu:0']
    else:
        dev_strs = raw_value.split(',')

    # framework
    raw_value = metafunc.config.getoption('--framework')
    if raw_value == 'all':
        f_strs = TEST_FRAMEWORKS.keys()
    else:
        f_strs = raw_value.split(',')

    # wrapped_mode
    raw_value = metafunc.config.getoption('--wrapped_mode')
    if raw_value == 'both':
        wrapped_modes = [True, False]
    elif raw_value:
        wrapped_modes = [True]
    else:
        wrapped_modes = [False]

    # create test configs
    configs = list()
    for f_str in f_strs:
        for dev_str in dev_strs:
            for wrapped_mode in wrapped_modes:
                configs.append((dev_str, TEST_FRAMEWORKS[f_str], wrapped_mode, TEST_CALL_METHODS[f_str]))
    metafunc.parametrize('dev_str,f,wrapped_mode,call', configs)


def pytest_addoption(parser):
    parser.addoption('--dev_str', action="store", default="cpu:0")
    parser.addoption('--framework', action="store", default="all")
    parser.addoption('--wrapped_mode', action="store", default="both")
