# local
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
from ivy.functional.frontends.paddle.tensor.math import log_softmax as paddle_log_softmax


tanh = paddle_tanh
log_softmax = paddle_log_softmax
