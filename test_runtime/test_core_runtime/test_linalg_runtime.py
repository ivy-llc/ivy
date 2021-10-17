"""
Collection of runtime tests for templated linalg functions
"""

DIM = int(1e4)


# global
import os
import random

# local
import ivy.core.general as ivy_gen
import ivy.core.linalg as ivy_linalg
this_file_dir = os.path.dirname(os.path.realpath(__file__))
import with_time_logs.ivy.core.linalg as ivy_linalg_w_time

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


def test_svd():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/linalg/svd.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[[random.uniform(0, 1), random.uniform(0, 1)],
                              [random.uniform(0, 1), random.uniform(0, 1)]] for _ in range(DIM)], f=lib)

        ivy_linalg.svd(x0, f=lib)
        ivy_linalg_w_time.svd(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_linalg_w_time.svd(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_linalg.svd(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_norm():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/linalg/norm.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[[random.uniform(0, 1), random.uniform(0, 1)],
                              [random.uniform(0, 1), random.uniform(0, 1)]] for _ in range(DIM)], f=lib)

        ivy_linalg.norm(x0, f=lib)
        ivy_linalg_w_time.norm(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_linalg_w_time.norm(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_linalg.norm(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_inv():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/linalg/inv.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[[random.uniform(0, 1), random.uniform(0, 1)],
                              [random.uniform(0, 1), random.uniform(0, 1)]] for _ in range(DIM)], f=lib)

        ivy_linalg.inv(x0, f=lib)
        ivy_linalg_w_time.inv(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_linalg_w_time.inv(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_linalg.inv(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_pinv():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/linalg/pinv.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[[random.uniform(0, 1), random.uniform(0, 1)],
                              [random.uniform(0, 1), random.uniform(0, 1)]] for _ in range(DIM)], f=lib)

        ivy_linalg.pinv(x0, f=lib)
        ivy_linalg_w_time.pinv(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_linalg_w_time.pinv(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_linalg.pinv(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_vector_to_skew_symmetric_matrix():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/linalg/vector_to_skew_symmetric_matrix.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                             for _ in range(DIM)], f=lib)

        ivy_linalg.vector_to_skew_symmetric_matrix(x0, f=lib)
        ivy_linalg_w_time.vector_to_skew_symmetric_matrix(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_linalg_w_time.vector_to_skew_symmetric_matrix(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_linalg.vector_to_skew_symmetric_matrix(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')
