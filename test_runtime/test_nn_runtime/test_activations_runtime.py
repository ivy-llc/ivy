"""
Collection of runtime tests for templated activation functions
"""

DIM = int(1e4)


# global
import os
import random

# local
import ivy.core.general as ivy_gen
import ivy.neural_net_functional.activations as ivy_act
this_file_dir = os.path.dirname(os.path.realpath(__file__))
import with_time_logs.ivy.neural_net.activations as ivy_act_w_time

from ivy import torch as _ivy_torch
from ivy import tensorflow as _ivy_tf
from ivy import mxnet as _ivy_mxnet
from ivy import jax as _ivy_jnp
from ivy import numpy as _ivy_np

from with_time_logs.ivy import torch as _ivy_torch_w_time
from with_time_logs.ivy import tensorflow as _ivy_tf_w_time
from with_time_logs.ivy import mxnet as _ivy_mxnet_w_time
from with_time_logs.ivy import jax as _ivy_jnp_w_time
from with_time_logs.ivy import numpy as _ivy_np_w_time

LIB_DICT = {_ivy_torch: _ivy_torch_w_time,
            _ivy_tf: _ivy_tf_w_time,
            _ivy_mxnet: _ivy_mxnet_w_time,
            _ivy_jnp: _ivy_jnp_w_time,
            _ivy_np: _ivy_np_w_time}

# local
import ivy_tests.helpers as helpers
from test_runtime.utils import append_to_file, log_time, write_times, TIMES_DICT


def test_relu():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/activations/relu.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_act.relu(x0, f=lib)
        ivy_act_w_time.relu(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_act_w_time.relu(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_act.relu(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_leaky_relu():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/activations/leaky_relu.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_act.leaky_relu(x0, f=lib)
        ivy_act_w_time.leaky_relu(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_act_w_time.leaky_relu(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_act.leaky_relu(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_tanh():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/activations/tanh.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_act.tanh(x0, f=lib)
        ivy_act_w_time.tanh(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_act_w_time.tanh(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_act.tanh(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_sigmoid():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/activations/sigmoid.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_act.sigmoid(x0, f=lib)
        ivy_act_w_time.sigmoid(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_act_w_time.sigmoid(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_act.sigmoid(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_softmax():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/activations/softmax.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_act.softmax(x0, f=lib)
        ivy_act_w_time.softmax(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_act_w_time.softmax(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_act.softmax(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_softplus():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/activations/softplus.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_act.softplus(x0, f=lib)
        ivy_act_w_time.softplus(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_act_w_time.softplus(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_act.softplus(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')
