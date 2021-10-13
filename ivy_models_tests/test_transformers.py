# global
import ivy
import pytest

# local
from ivy_models.transformers.helpers import FeedForward, PreNorm
from ivy_models.transformers.perceiver import PerceiverSpec, Perceiver
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


# Perceiver #
# ----------#

@pytest.mark.parametrize(
    "batch_size", [3])
@pytest.mark.parametrize(
    "img_dims", [[16, 16]])
@pytest.mark.parametrize(
    "output_dim", [10])
def test_perceiver_img_classification(dev_str, f, call, batch_size, img_dims, output_dim):
    model = Perceiver(PerceiverSpec(input_dim=3, num_input_axes=2, output_dim=output_dim,
                                    max_fourier_freq=img_dims[0], device=dev_str))
    img = ivy.random_uniform(shape=[batch_size] + img_dims + [3], dev_str=dev_str)
    ret = model(img)
    assert list(ret.shape) == [batch_size, output_dim]


# Perceiver IO #
# -------------#

@pytest.mark.parametrize(
    "batch_size", [1])
@pytest.mark.parametrize(
    "img_dims", [[16, 16]])
@pytest.mark.parametrize(
    "queries_dim", [32])
@pytest.mark.parametrize(
    "learn_query", [True, False])
def test_perceiver_io_img_classification(dev_str, f, call, batch_size, img_dims, queries_dim, learn_query):

    # params
    input_dim = 3
    num_input_axes = 2
    output_dim = 10

    # inputs
    img = ivy.random_uniform(shape=[batch_size] + img_dims + [3], dev_str=dev_str)
    queries = None if learn_query else ivy.random_uniform(shape=[batch_size, 1, queries_dim], dev_str=dev_str)

    # model call
    model = PerceiverIO(PerceiverIOSpec(input_dim=input_dim,
                                        num_input_axes=num_input_axes,
                                        output_dim=output_dim,
                                        queries_dim=queries_dim,
                                        learn_query=learn_query,
                                        query_shape=[1],
                                        max_fourier_freq=img_dims[0],
                                        device=dev_str))

    # output
    output = model(img, queries=queries)

    # cardinality test
    assert output.shape == (batch_size, 1, output_dim)
