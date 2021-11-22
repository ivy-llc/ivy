# global
import os
import ivy
import pytest
import numpy as np

# local
from ivy_models.transformers.helpers import FeedForward, PreNorm
from ivy_models.transformers.perceiver_io import PerceiverIOSpec, PerceiverIO


# Helpers #
# --------#

def test_feedforward(dev_str, f, call):
    ivy.seed(0)
    feedforward = FeedForward(4, dev_str=dev_str)
    x = ivy.random_uniform(shape=(1, 3, 4), dev_str=dev_str)
    ret = feedforward(x)
    assert list(ret.shape) == [1, 3, 4]


def test_prenorm(dev_str, f, call):
    ivy.seed(0)
    att = ivy.MultiHeadAttention(4, dev_str=dev_str)
    prenorm = PreNorm(4, att, dev_str=dev_str)
    x = ivy.random_uniform(shape=(1, 3, 4), dev_str=dev_str)
    ret = prenorm(x)
    assert list(ret.shape) == [1, 3, 4]


# Perceiver IO #
# -------------#

@pytest.mark.parametrize(
    "batch_shape", [[1]])
@pytest.mark.parametrize(
    "img_dims", [[224, 224]])
@pytest.mark.parametrize(
    "queries_dim", [1024])
@pytest.mark.parametrize(
    "learn_query", [True])
@pytest.mark.parametrize(
    "load_weights", [True, False])
def test_perceiver_io_img_classification(dev_str, f, call, batch_shape, img_dims, queries_dim, learn_query,
                                         load_weights):
    # params
    input_dim = 3
    num_input_axes = 2
    output_dim = 10

    # inputs
    this_dir = os.path.dirname(os.path.realpath(__file__))
    img = ivy.array(np.load(os.path.join(this_dir, 'img.npy'))[None], dtype_str='float32', dev_str=dev_str)
    queries = None if learn_query else ivy.random_uniform(shape=batch_shape + [1, queries_dim], dev_str=dev_str)

    model = PerceiverIO(PerceiverIOSpec(input_dim=input_dim,
                                        num_input_axes=num_input_axes,
                                        output_dim=output_dim,
                                        queries_dim=queries_dim,
                                        learn_query=learn_query,
                                        query_shape=[1],
                                        max_fourier_freq=img_dims[0],
                                        num_fourier_freq_bands=64,
                                        device=dev_str))

    # maybe load weights
    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(this_dir, '../ivy_models/transformers/pretrained_weights/perceiver_io.hdf5')
        v = ivy.Container.from_disk_as_hdf5(weight_fpath)

        # try:
        #     assert model.v.num_arrays() == v.num_arrays()
        #     assert ivy.Container.identical_array_shapes([model.v, v])
        # except AssertionError:
        #     raise Exception(
        #         'model.v.size_ordered_arrays(): {}\n\n'
        #         'v.size_ordered_arrays(): {}\n\n'.format(
        #             model.v.size_ordered_arrays(), v.size_ordered_arrays()))

        # ToDo: incrementally update this restructuring, so that the loaded jax weights are converted
        v = v.restructure_key_chains(
            {'perceiver_encoder/~/trainable_position_encoding/pos_embs': 'latents',
             'perceiver_encoder/~/cross_attention/layer_norm/scale': 'layers/v0/cross_att/norm/scale',
             'perceiver_encoder/~/cross_attention/layer_norm/offset': 'layers/v0/cross_att/norm/offset'},
            keep_orig=False)

        # assert ivy.Container.identical_structure([model.v, v])

        model = PerceiverIO(PerceiverIOSpec(input_dim=input_dim,
                                            num_input_axes=num_input_axes,
                                            output_dim=output_dim,
                                            queries_dim=queries_dim,
                                            learn_query=learn_query,
                                            query_shape=[1],
                                            max_fourier_freq=img_dims[0],
                                            device=dev_str), v=v, with_partial_v=True)

        # expected submodule returns
        expected_submod_rets = ivy.Container()
        for key in ['LayerNorm_0']:
            expected_submod_rets[key] = {'val': np.load(os.path.join(this_dir, key + '.npy')),
                                         'atol': 1e-6, 'rtol': 1e-6}

        # check submod returns
        output = model(img, queries=queries, expected_submod_rets=expected_submod_rets)

    else:

        # output
        output = model(img, queries=queries)

    # cardinality test
    assert output.shape == tuple(batch_shape + [1, output_dim])


@pytest.mark.parametrize(
    "batch_shape", [[3]])
@pytest.mark.parametrize(
    "img_dims", [[32, 32]])
@pytest.mark.parametrize(
    "queries_dim", [32])
@pytest.mark.parametrize(
    "learn_query", [True, False])
def test_perceiver_io_flow_prediction(dev_str, f, call, batch_shape, img_dims, queries_dim, learn_query):
    # params
    input_dim = 3
    num_input_axes = 3
    output_dim = 2

    # inputs
    img = ivy.random_uniform(shape=batch_shape + [2] + img_dims + [3], dev_str=dev_str)
    queries = ivy.random_uniform(shape=batch_shape + img_dims + [32], dev_str=dev_str)

    # model call
    model = PerceiverIO(PerceiverIOSpec(input_dim=input_dim,
                                        num_input_axes=num_input_axes,
                                        output_dim=output_dim,
                                        queries_dim=queries_dim,
                                        learn_query=learn_query,
                                        query_shape=img_dims,
                                        max_fourier_freq=img_dims[0],
                                        device=dev_str))

    # output
    output = model(img, queries=queries)

    # cardinality test
    assert output.shape == tuple(batch_shape + img_dims + [output_dim])
