"""
Collection of runtime tests for templated layers functions
"""

DIM = int(1e4)


# global
import os
import random

# local
import ivy.core.general as ivy_gen
import ivy.neural_net_stateful.layers as ivy_layers
this_file_dir = os.path.dirname(os.path.realpath(__file__))
import with_time_logs.ivy.neural_net.layers as ivy_layers_w_time

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

LIB_DICT = {_ivy_torch: _ivy_torch_w_time,
            _ivy_tf: _ivy_tf_w_time,
            _ivy_mxnd: _ivy_mxnd_w_time,
            _ivy_jnp: _ivy_jnp_w_time,
            _ivy_np: _ivy_np_w_time}

# local
import ivy_tests.helpers as helpers
from test_runtime.utils import append_to_file, log_time, write_times, TIMES_DICT


def test_conv1d():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/layers/conv1d.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 2d convolutions
            continue

        x0 = ivy_gen.tensor([[[random.uniform(0, 1)], [random.uniform(0, 1)]] for _ in range(DIM)], f=lib)
        if call in [helpers.tf_call, helpers.tf_graph_call]:
            data_format = "NWC"
        else:
            x0 = ivy_gen.reshape(x0, (DIM, 1, 2))
            data_format = "NCW"

        append_to_file(fname, '{}'.format(lib))

        filters = ivy_gen.tensor([[[0.]], [[1.]]], f=lib)
        ivy_layers.conv1d(x0, filters, 1, "SAME", data_format, filter_shape=[2], num_filters=1, f=lib)
        ivy_layers_w_time.conv1d(x0, filters, 1, "SAME", data_format, filter_shape=[2], num_filters=1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_layers_w_time.conv1d(x0, filters, 1, "SAME", data_format, filter_shape=[2], num_filters=1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_layers.conv1d(x0, filters, 1, "SAME", data_format, filter_shape=[2], num_filters=1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_conv1d_transpose():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/layers/conv1d_transpose.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 2d convolutions
            continue

        x0 = ivy_gen.tensor([[[random.uniform(0, 1)], [random.uniform(0, 1)]] for _ in range(DIM)], f=lib)
        if call in [helpers.tf_call, helpers.tf_graph_call]:
            data_format = "NWC"
        else:
            x0 = ivy_gen.reshape(x0, (DIM, 1, 2))
            data_format = "NCW"

        append_to_file(fname, '{}'.format(lib))

        filters = ivy_gen.tensor([[[0.]], [[1.]]], f=lib)
        ivy_layers.conv1d_transpose(x0, filters, 1, "SAME", (DIM, 2, 1), data_format, filter_shape=[2], num_filters=1,
                                    f=lib)
        ivy_layers_w_time.conv1d_transpose(x0, filters, 1, "SAME", (DIM, 2, 1), data_format, filter_shape=[2],
                                           num_filters=1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_layers_w_time.conv1d_transpose(x0, filters, 1, "SAME", (DIM, 2, 1), data_format, filter_shape=[2],
                                               num_filters=1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_layers.conv1d_transpose(x0, filters, 1, "SAME", (DIM, 2, 1), data_format, filter_shape=[2],
                                        num_filters=1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_conv2d():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/layers/conv2d.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 2d convolutions
            continue

        x0 = ivy_gen.tensor([[[[random.uniform(0, 1)], [random.uniform(0, 1)]],
                              [[random.uniform(0, 1)], [random.uniform(0, 1)]]] for _ in range(DIM)], f=lib)
        if call in [helpers.tf_call, helpers.tf_graph_call]:
            data_format = "NHWC"
        else:
            x0 = ivy_gen.reshape(x0, (DIM, 1, 2, 2))
            data_format = "NCHW"

        append_to_file(fname, '{}'.format(lib))

        filters = ivy_gen.tensor([[[[0.]], [[1.]]],
                                  [[[1.]], [[0.]]]], f=lib)
        ivy_layers.conv2d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2], num_filters=1, f=lib)
        ivy_layers_w_time.conv2d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2], num_filters=1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_layers_w_time.conv2d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2], num_filters=1,
                                     f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_layers.conv2d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2], num_filters=1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_conv2d_transpose():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/layers/conv2d_transpose.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 2d convolutions
            continue

        x0 = ivy_gen.tensor([[[[random.uniform(0, 1)], [random.uniform(0, 1)]],
                              [[random.uniform(0, 1)], [random.uniform(0, 1)]]] for _ in range(DIM)], f=lib)
        if call in [helpers.tf_call, helpers.tf_graph_call]:
            data_format = "NHWC"
        else:
            x0 = ivy_gen.reshape(x0, (DIM, 1, 2, 2))
            data_format = "NCHW"

        append_to_file(fname, '{}'.format(lib))

        filters = ivy_gen.tensor([[[[0.]], [[1.]]],
                                  [[[1.]], [[0.]]]], f=lib)
        ivy_layers.conv2d_transpose(x0, filters, 1, "SAME", (DIM, 2, 2, 1), data_format, filter_shape=[2, 2],
                                    num_filters=1, f=lib)
        ivy_layers_w_time.conv2d_transpose(x0, filters, 1, "SAME", (DIM, 2, 2, 1), data_format, filter_shape=[2, 2],
                                           num_filters=1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_layers_w_time.conv2d_transpose(x0, filters, 1, "SAME", (DIM, 2, 2, 1), data_format, filter_shape=[2, 2],
                                               num_filters=1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_layers.conv2d_transpose(x0, filters, 1, "SAME", (DIM, 2, 2, 1), data_format, filter_shape=[2, 2],
                                        num_filters=1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_depthwise_conv2d():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/layers/depthwise_conv2d.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 2d convolutions
            continue

        x0 = ivy_gen.tensor([[[[random.uniform(0, 1)], [random.uniform(0, 1)]],
                              [[random.uniform(0, 1)], [random.uniform(0, 1)]]] for _ in range(DIM)], f=lib)
        if call in [helpers.tf_call, helpers.tf_graph_call]:
            data_format = "NHWC"
        else:
            x0 = ivy_gen.reshape(x0, (DIM, 1, 2, 2))
            data_format = "NCHW"

        append_to_file(fname, '{}'.format(lib))

        filters = ivy_gen.tensor([[[0.], [1.]],
                                  [[1.], [0.]]], f=lib)
        ivy_layers.depthwise_conv2d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2], num_filters=1,
                                    num_channels=1, f=lib)
        ivy_layers_w_time.depthwise_conv2d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2], num_filters=1,
                                           f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_layers_w_time.depthwise_conv2d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2], num_filters=1,
                                               f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_layers.depthwise_conv2d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2], num_filters=1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_conv3d():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/layers/conv3d.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        if call in [helpers.np_call, helpers.jnp_call]:
            # numpy and jax do not yet support 3d convolutions
            continue

        x0 = ivy_gen.tensor([[[[[random.uniform(0, 1)], [random.uniform(0, 1)]],
                               [[random.uniform(0, 1)], [random.uniform(0, 1)]]],
                              [[[random.uniform(0, 1)], [random.uniform(0, 1)]],
                              [[random.uniform(0, 1)], [random.uniform(0, 1)]]]] for _ in range(DIM)], f=lib)
        if call in [helpers.tf_call, helpers.tf_graph_call]:
            data_format = "NDHWC"
        else:
            x0 = ivy_gen.reshape(x0, (DIM, 1, 2, 2, 2))
            data_format = "NCDHW"

        append_to_file(fname, '{}'.format(lib))

        filters = ivy_gen.tensor([[[[[0.]], [[0.]]], [[[1.]], [[1.]]]],
                                  [[[[1.]], [[1.]]], [[[0.]], [[0.]]]]], f=lib)
        ivy_layers.conv3d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2, 2], num_filters=1, f=lib)
        ivy_layers_w_time.conv3d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2, 2], num_filters=1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_layers_w_time.conv3d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2, 2], num_filters=1,
                                     f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_layers.conv3d(x0, filters, 1, "SAME", data_format, filter_shape=[2, 2, 2], num_filters=1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_conv3d_transpose():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/layers/conv3d_transpose.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call, helpers.mx_graph_call]:
            # numpy and jax do not yet support 3d convolutions, and mxnet only supports with CUDNN
            continue

        x0 = ivy_gen.tensor([[[[[random.uniform(0, 1)], [random.uniform(0, 1)]],
                               [[random.uniform(0, 1)], [random.uniform(0, 1)]]],
                              [[[random.uniform(0, 1)], [random.uniform(0, 1)]],
                              [[random.uniform(0, 1)], [random.uniform(0, 1)]]]] for _ in range(DIM)], f=lib)
        if call in [helpers.tf_call, helpers.tf_graph_call]:
            data_format = "NDHWC"
        else:
            x0 = ivy_gen.reshape(x0, (DIM, 1, 2, 2, 2))
            data_format = "NCDHW"

        append_to_file(fname, '{}'.format(lib))

        filters = ivy_gen.tensor([[[[[0.]], [[0.]]], [[[1.]], [[1.]]]],
                                  [[[[1.]], [[1.]]], [[[0.]], [[0.]]]]], f=lib)
        ivy_layers.conv3d_transpose(x0, filters, 1, "SAME", (DIM, 2, 2, 2, 1), data_format, filter_shape=[2, 2, 2],
                                    num_filters=1, f=lib)
        ivy_layers_w_time.conv3d_transpose(x0, filters, 1, "SAME", (DIM, 2, 2, 2, 1), data_format,
                                           filter_shape=[2, 2, 2], num_filters=1, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_layers_w_time.conv3d_transpose(x0, filters, 1, "SAME", (DIM, 2, 2, 2, 1), data_format,
                                               filter_shape=[2, 2, 2], num_filters=1, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_layers.conv3d_transpose(x0, filters, 1, "SAME", (DIM, 2, 2, 2, 1), data_format, filter_shape=[2, 2, 2],
                                        num_filters=1, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')


def test_linear():

    fname = os.path.join(this_file_dir, 'runtime_analysis/{}/layers/linear.txt'.format(DIM))
    if os.path.exists(fname):
        os.remove(fname)
    for lib, call in [(l, c) for l, c in helpers.calls if c not in [helpers.tf_graph_call, helpers.mx_graph_call]]:

        time_lib = LIB_DICT[lib]

        append_to_file(fname, '{}'.format(lib))

        x0 = ivy_gen.tensor([[random.uniform(0, 1) for _ in range(DIM)]], f=lib)
        weight = ivy_gen.tensor([[random.uniform(0, 1) for _ in range(DIM)],
                                 [random.uniform(0, 1) for _ in range(DIM)]], f=lib)
        bias = ivy_gen.tensor([random.uniform(0, 1), random.uniform(0, 1)], f=lib)

        ivy_layers.linear(x0, weight, bias, num_hidden=2, f=lib)
        ivy_layers_w_time.linear(x0, weight, bias, num_hidden=2, f=time_lib)
        TIMES_DICT.clear()

        for _ in range(100):

            log_time(fname, 'tb0')
            ivy_layers_w_time.linear(x0, weight, bias, num_hidden=2, f=time_lib)
            log_time(fname, 'tb4', time_at_start=True)

            log_time(fname, 'tt0')
            ivy_layers.linear(x0, weight, bias, num_hidden=2, f=lib)
            log_time(fname, 'tt1', time_at_start=True)

        write_times()

    append_to_file(fname, 'end of analysis')
