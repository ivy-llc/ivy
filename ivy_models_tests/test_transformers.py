# global
import ivy

# local
from ivy_models.transformers.perceiver import FeedForward, PreNorm, PerceiverSpec, Perceiver


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


def test_perceiver_img_input(dev_str, f, call):
    model = Perceiver(PerceiverSpec(num_input_dims=3, num_input_axes=2, num_classes=1000, max_fourier_freq=224))
    img = ivy.random_uniform(shape=(1, 224, 224, 3))  # i.e. 1 imagenet image
    ret = model(img)
    assert list(ret.shape) == [1, 1000]
