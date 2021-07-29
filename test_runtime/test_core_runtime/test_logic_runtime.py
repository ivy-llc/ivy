"""
Collection of runtime tests for templated logic functions
"""

# global
import os
import random

# local
import ivy.core.general as ivy_gen
import ivy.core.logic as ivy_logic
import with_time_logs.ivy.core.logic as ivy_logic_w_time

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

DIM = int(1e4)

LIB_DICT = {
    _ivy_torch: _ivy_torch_w_time,
    _ivy_tf: _ivy_tf_w_time,
    _ivy_mxnd: _ivy_mxnd_w_time,
    _ivy_jnp: _ivy_jnp_w_time,
    _ivy_np: _ivy_np_w_time
}

# local
import ivy_tests.helpers as helpers
from test_runtime.utils import append_to_file, log_time, write_times, TIMES_DICT


this_file_dir = os.path.dirname(os.path.realpath(__file__))


def test_logical_and():
    fname = os.path.join(
        this_file_dir,
        'runtime_analysis/{}/logic/logical_and.txt'.format(DIM)
    )

    if os.path.exists(fname):
        os.remove(fname)

    for lib, call in [
        (l, c) for l, c in helpers.calls
        if c not in [helpers.tf_graph_call, helpers.mx_graph_call]
    ]:

        time_lib = LIB_DICT[lib]
        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor(
            [random.uniform(0, 1) for _ in range(DIM)], f=lib
        ) > 0.3

        x1 = ivy_gen.tensor(
            [random.uniform(0, 1) for _ in range(DIM)], f=lib
        ) > 0.6

        ivy_logic.logical_and(x0, x1, f=lib)
        ivy_logic_w_time.logical_and(x0, x1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):
            log_time(fname, 'tb0')
            ivy_logic_w_time.logical_and(x0, x1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_logic.logical_and(x0, x1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_logical_or():
    fname = os.path.join(
        this_file_dir,
        'runtime_analysis/{}/logic/logical_or.txt'.format(DIM)
    )

    if os.path.exists(fname):
        os.remove(fname)

    for lib, call in [
        (l, c) for l, c in helpers.calls
        if c not in [helpers.tf_graph_call, helpers.mx_graph_call]
    ]:

        time_lib = LIB_DICT[lib]
        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib) > 0.3
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib) > 0.6

        ivy_logic.logical_or(x0, x1, f=lib)
        ivy_logic_w_time.logical_or(x0, x1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):
            log_time(fname, 'tb0')
            ivy_logic_w_time.logical_or(x0, x1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_logic.logical_or(x0, x1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_logical_not():
    fname = os.path.join(
        this_file_dir,
        'runtime_analysis/{}/logic/logical_not.txt'.format(DIM)
    )

    if os.path.exists(fname):
        os.remove(fname)

    for lib, call in [
        (l, c) for l, c in helpers.calls
        if c not in [helpers.tf_graph_call, helpers.mx_graph_call]
    ]:

        time_lib = LIB_DICT[lib]
        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor(
            [random.uniform(0, 1) for _ in range(DIM)], f=lib
        ) > 0.5

        ivy_logic.logical_not(x0, f=lib)
        ivy_logic_w_time.logical_not(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):
            log_time(fname, 'tb0')
            ivy_logic_w_time.logical_not(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_logic.logical_not(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')
