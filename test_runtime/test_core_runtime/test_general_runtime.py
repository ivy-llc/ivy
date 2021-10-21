"""
Collection of runtime tests for templated general functions
"""

DIM = int(1e4)


# global
import os
import random

# local
import ivy.core.general as ivy_gen
this_file_dir = os.path.dirname(os.path.realpath(__file__))
import with_time_logs.ivy.core.general as ivy_gen_w_time

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


def test_array():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/array.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = [random.uniform(0, 1) for _ in range(DIM)]

        ivy_gen.tensor(x0, f=lib)
        ivy_gen_w_time.tensor(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.tensor(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.tensor(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_to_numpy():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/to_numpy.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.to_numpy(x0, f=lib)
        ivy_gen_w_time.to_numpy(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.to_numpy(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.to_numpy(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_to_list():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/to_list.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.to_list(x0, f=lib)
        ivy_gen_w_time.to_list(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.to_list(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.to_list(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_shape():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/shape.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.shape(x0, f=lib)
        ivy_gen_w_time.shape(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.shape(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.shape(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_get_num_dims():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/get_num_dims.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.get_num_dims(x0, f=lib)
        ivy_gen_w_time.get_num_dims(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.get_num_dims(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.get_num_dims(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_minimum():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/minimum.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.minimum(x0, x1, f=lib)
        ivy_gen_w_time.minimum(x0, x1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.minimum(x0, x1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.minimum(x0, x1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_maximum():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/maximum.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.maximum(x0, x1, f=lib)
        ivy_gen_w_time.maximum(x0, x1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.maximum(x0, x1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.maximum(x0, x1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_clip():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/clip.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.clip(x0, 0, 1, f=lib)
        ivy_gen_w_time.clip(x0, 0, 1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.clip(x0, 0, 1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.clip(x0, 0, 1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_round():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/round.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.round(x0, f=lib)
        ivy_gen_w_time.round(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.round(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.round(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_floormod():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/floormod.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.floormod(x0, x1, f=lib)
        ivy_gen_w_time.floormod(x0, x1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.floormod(x0, x1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.floormod(x0, x1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_floor():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/floor.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.floor(x0, f=lib)
        ivy_gen_w_time.floor(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.floor(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.floor(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_ceil():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/ceil.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.ceil(x0, f=lib)
        ivy_gen_w_time.ceil(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.ceil(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.ceil(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_abs():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/abs.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.abs(x0, f=lib)
        ivy_gen_w_time.abs(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.abs(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.abs(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_argmax():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/argmax.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.argmax(x0, f=lib)
        ivy_gen_w_time.argmax(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.argmax(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.argmax(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_argmin():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/argmin.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.argmin(x0, f=lib)
        ivy_gen_w_time.argmin(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.argmin(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.argmin(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_cast():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/cast.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.cast(x0, 'float32', f=lib)
        ivy_gen_w_time.cast(x0, 'float32', f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.cast(x0, 'float32', f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.cast(x0, 'float32', f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_arange():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/arange.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        ivy_gen.arange(DIM, f=lib)
        ivy_gen_w_time.arange(DIM, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.arange(DIM, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.arange(DIM, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_linspace():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/linspace.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        ivy_gen.linspace(0, DIM, DIM, f=lib)
        ivy_gen_w_time.linspace(0, DIM, DIM, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.linspace(0, DIM, DIM, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.linspace(0, DIM, DIM, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_concatenate():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/concatenate.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.concatenate([x0, x1], f=lib)
        ivy_gen_w_time.concatenate(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.concatenate([x0, x1], 0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.concatenate([x0, x1], 0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_flip():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/flip.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.flip(x0, 0, f=lib)
        ivy_gen_w_time.flip(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.flip(x0, 0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.flip(x0, 0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_stack():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/stack.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.stack([x0, x1], f=lib)
        ivy_gen_w_time.stack([x0, x1], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.stack([x0, x1], 0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.stack([x0, x1], 0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_unstack():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/unstack.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.uniform(0, 1) for _ in range(DIM)]], f=lib)

        ivy_gen.unstack(x0, 0, num_outputs=1, f=lib)
        ivy_gen_w_time.unstack(x0, 0, num_outputs=1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.unstack(x0, 0, num_outputs=1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.unstack(x0, 0, num_outputs=1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_split():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/split.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.uniform(0, 1) for _ in range(DIM)]], f=lib)

        ivy_gen.split(x0, 1, 0, f=lib)
        ivy_gen_w_time.split(x0, 1, 0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.split(x0, 1, 0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.split(x0, 1, 0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_tile():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/tile.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1)], f=lib)

        ivy_gen.tile(x0, [DIM], f=lib)
        ivy_gen_w_time.tile(x0, [DIM], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.tile(x0, [DIM], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.tile(x0, [DIM], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_zero_pad():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/zero_pad.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.zero_pad(x0, [(DIM, DIM)], x_shape=[DIM], f=lib)
        ivy_gen_w_time.zero_pad(x0, [(DIM, DIM)], x_shape=[DIM], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.zero_pad(x0, [(DIM, DIM)], x_shape=[DIM], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.zero_pad(x0, [(DIM, DIM)], x_shape=[DIM], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_swapaxes():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/swapaxes.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.uniform(0, 1) for _ in range(DIM)], [random.uniform(0, 1) for _ in range(DIM)]], f=lib)

        ivy_gen.swapaxes(x0, 1, 0, f=lib)
        ivy_gen_w_time.swapaxes(x0, 1, 0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.swapaxes(x0, 1, 0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.swapaxes(x0, 1, 0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_transpose():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/transpose.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.uniform(0, 1) for _ in range(DIM)], [random.uniform(0, 1) for _ in range(DIM)]], f=lib)

        ivy_gen.transpose(x0, (1, 0), f=lib)
        ivy_gen_w_time.transpose(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.transpose(x0, (1, 0), f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.transpose(x0, (1, 0), f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_expand_dims():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/expand_dims.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.expand_dims(x0, 0, f=lib)
        ivy_gen_w_time.expand_dims(x0, 0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.expand_dims(x0, 0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.expand_dims(x0, 0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_where():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/where.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib) > 0
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)
        x2 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.where(x0, x1, x2, condition_shape=[DIM], x_shape=[DIM], f=lib)
        ivy_gen_w_time.where(x0, x1, x2, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.where(x0, x1, x2, condition_shape=[DIM], x_shape=[DIM], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.where(x0, x1, x2, condition_shape=[DIM], x_shape=[DIM], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_indices_where():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/indices_where.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(-1, 1) for _ in range(DIM)], f=lib) > 0

        ivy_gen.indices_where(x0, f=lib)
        ivy_gen_w_time.indices_where(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.indices_where(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.indices_where(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_reshape():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/reshape.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.reshape(x0, [1, DIM], f=lib)
        ivy_gen_w_time.reshape(x0, [1, DIM], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.reshape(x0, [1, DIM], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.reshape(x0, [1, DIM], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_squeeze():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/squeeze.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.uniform(0, 1) for _ in range(DIM)]], f=lib)

        ivy_gen.squeeze(x0, f=lib)
        ivy_gen_w_time.squeeze(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.squeeze(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.squeeze(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_zeros():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/zeros.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        ivy_gen.zeros([DIM], f=lib)
        ivy_gen_w_time.zeros([DIM], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.zeros([DIM], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.zeros([DIM], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_zeros_like():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/zeros_like.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.zeros_like(x0, f=lib)
        ivy_gen_w_time.zeros_like(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.zeros_like(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.zeros_like(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_ones():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/ones.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        ivy_gen.ones([DIM], f=lib)
        ivy_gen_w_time.ones([DIM], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.ones([DIM], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.ones([DIM], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_ones_like():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/ones_like.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.ones_like(x0, f=lib)
        ivy_gen_w_time.ones_like(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.ones_like(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.ones_like(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_one_hot():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/one_hot.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.cast(ivy_gen.tensor([random.randint(0, int(DIM ** 0.5) - 1)
                                          for _ in range(int(DIM**0.5))], f=lib), 'int64')

        ivy_gen.one_hot(x0, int(DIM**0.5), f=lib)
        ivy_gen_w_time.one_hot(x0, int(DIM**0.5), f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.one_hot(x0, int(DIM**0.5), f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.one_hot(x0, int(DIM**0.5), f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_cross():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/cross.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.uniform(0, 1)] * 3 for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([[random.uniform(0, 1)] * 3 for _ in range(DIM)], f=lib)

        ivy_gen.cross(x0, x1, f=lib)
        ivy_gen_w_time.cross(x0, x1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.cross(x0, x1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.cross(x0, x1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_matmul():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/matmul.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[[random.uniform(0, 1) for _ in range(DIM)]]], f=lib)
        x1 = ivy_gen.tensor([[[random.uniform(0, 1)] for _ in range(DIM)]], f=lib)

        ivy_gen.matmul(x0, x1, batch_shape=[1], f=lib)
        ivy_gen_w_time.matmul(x0, x1, batch_shape=[1], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.matmul(x0, x1, batch_shape=[1], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.matmul(x0, x1, batch_shape=[1], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_cumsum():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/cumsum.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.cumsum(x0, 0, f=lib)
        ivy_gen_w_time.cumsum(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.cumsum(x0, 0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.cumsum(x0, 0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_identity():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/identity.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        ivy_gen.identity(int(DIM**0.5), f=lib)
        ivy_gen_w_time.identity(int(DIM**0.5), f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.identity(int(DIM**0.5), f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.identity(int(DIM**0.5), f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_scatter_flat():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/scatter_flat.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.randint(0, DIM - 1) for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.scatter_flat(x0, x1, DIM, f=lib)
        ivy_gen_w_time.scatter_flat(x0, x1, DIM, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.scatter_flat(x0, x1, DIM, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.scatter_flat(x0, x1, DIM, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_scatter_nd():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/scatter_nd.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.randint(0, DIM - 1)] for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.scatter_nd(x0, x1, [DIM], f=lib)
        ivy_gen_w_time.scatter_nd(x0, x1, [DIM], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.scatter_nd(x0, x1, [DIM], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.scatter_nd(x0, x1, [DIM], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_gather_flat():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/gather_flat.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([random.randint(0, DIM - 1) for _ in range(DIM)], f=lib)

        ivy_gen.gather(x0, x1, f=lib)
        ivy_gen_w_time.gather(x0, x1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.gather(x0, x1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.gather(x0, x1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_gather_nd():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/gather_nd.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)
        x1 = ivy_gen.tensor([[random.randint(0, DIM - 1)] for _ in range(DIM)], f=lib)

        ivy_gen.gather_nd(x0, x1, indices_shape=[DIM, 1], f=lib)
        ivy_gen_w_time.gather_nd(x0, x1, indices_shape=[DIM, 1], f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.gather_nd(x0, x1, indices_shape=[DIM, 1], f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.gather_nd(x0, x1, indices_shape=[DIM, 1], f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_get_device():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/get_device.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.dev_str(x0, f=lib)
        ivy_gen_w_time.dev_str(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.dev_str(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.dev_str(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_dtype():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/dtype.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.dtype(x0, f=lib)
        ivy_gen_w_time.dtype(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.dtype(x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.dtype(x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_compile_backend():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/general/compile_backend.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        if call is helpers.mx_call:
            continue

        append_to_file(fname, '{}'.format(lib))

        some_fn = lambda x: x ** 2
        x0 = ivy_gen.tensor([random.uniform(0, 1) for _ in range(DIM)], f=lib)

        ivy_gen.dtype(x0, f=lib)
        ivy_gen_w_time.dtype(x0, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_gen_w_time.compile_backend(some_fn, x0, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_gen.compile_backend(some_fn, x0, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')
