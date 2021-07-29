"""
Collection of runtime tests for templated random functions
"""

# global
import os
import random

# local
import ivy.core.general as ivy_gen
import ivy.core.random as ivy_rand

import with_time_logs.ivy.core.random as ivy_rand_w_time

from ivy import torch as _ivy_torch
from ivy import tensorflow as _ivy_tf
from ivy import mxnd as _ivy_mxnd
from ivy import jax as _ivy_jnp
from ivy import numpy as _ivy_np

from with_time_logs.ivy import torch as _ivy_torch_w_time
from with_time_logs.ivy import tensorflow as _ivy_tf_w_time
from with_time_logs.ivy import mxnd as _ivy_mxnd_w_time
from with_time_logs.ivy import jax as _ivy_jnp_w_time
from with_time_logs.ivy import numpy as _ivy_np_w_time

# local
import ivy_tests.helpers as helpers
from test_runtime.utils import append_to_file, log_time, write_times, TIMES_DICT

DIM = int(1e4)

LIB_DICT = {
    _ivy_torch: _ivy_torch_w_time,
    _ivy_tf: _ivy_tf_w_time,
    _ivy_mxnd: _ivy_mxnd_w_time,
    _ivy_jnp: _ivy_jnp_w_time,
    _ivy_np: _ivy_np_w_time
}

this_file_dir = os.path.dirname(os.path.realpath(__file__))


def test_random_uniform():
    fname = os.path.join(
        this_file_dir,
        'runtime_analysis/{}/random/random_uniform.txt'.format(DIM)
    )

    if os.path.exists(fname):
        os.remove(fname)

    for lib, call in [
        (l, c) for l, c in helpers.calls
        if c not in [helpers.tf_graph_call, helpers.mx_graph_call]
    ]:

        time_lib = LIB_DICT[lib]
        append_to_file(fname, '{}'.format(lib))

        ivy_rand.random_uniform(0, 1, (DIM,), f=lib)
        ivy_rand_w_time.random_uniform(0, 1, (DIM,), f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):
            log_time(fname, 'tb0')
            ivy_rand_w_time.random_uniform(0, 1, (DIM,), f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_rand.random_uniform(0, 1, (DIM,), f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_randint():
    fname = os.path.join(
        this_file_dir, 'runtime_analysis/{}/random/randint.txt'.format(DIM)
    )

    if os.path.exists(fname):
        os.remove(fname)

    for lib, call in [
        (l, c) for l, c in helpers.calls
        if c not in [helpers.tf_graph_call, helpers.mx_graph_call]
    ]:

        time_lib = LIB_DICT[lib]
        append_to_file(fname, '{}'.format(lib))

        ivy_rand.randint(0, 10, (DIM,), f=lib)
        ivy_rand_w_time.randint(0, 10, (DIM,), f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):
            log_time(fname, 'tb0')
            ivy_rand_w_time.randint(0, 10, (DIM,), f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_rand.randint(0, 10, (DIM,), f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_seed():
    fname = os.path.join(
        this_file_dir, 'runtime_analysis/{}/random/seed.txt'.format(DIM)
    )

    if os.path.exists(fname):
        os.remove(fname)

    for lib, call in [
        (l, c) for l, c in helpers.calls
        if c not in [helpers.tf_graph_call, helpers.mx_graph_call]
    ]:

        time_lib = LIB_DICT[lib]
        append_to_file(fname, '{}'.format(lib))

        ivy_rand.seed(10, f=lib)
        ivy_rand_w_time.seed(10, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):
            log_time(fname, 'tb0')
            ivy_rand_w_time.seed(_, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_rand.seed(_, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_shuffle():
    fname = os.path.join(
        this_file_dir, 'runtime_analysis/{}/random/shuffle.txt'.format(DIM)
    )

    if os.path.exists(fname):
        os.remove(fname)

    for lib, call in [
        (l, c) for l, c in helpers.calls
        if c not in [helpers.tf_graph_call, helpers.mx_graph_call]
    ]:

        time_lib = LIB_DICT[lib]
        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_rand.shuffle(x0, f=lib)
        ivy_rand_w_time.shuffle(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):
            log_time(fname, 'tb0')
            ivy_rand_w_time.shuffle(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_rand.shuffle(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')
