# local
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
import ivy


tanh = paddle_tanh


def softplus(x, beta=1, threshold=20):
    return ivy.where(x * beta > threshold, x, ivy.log(1 + ivy.exp(x * beta)) / beta)