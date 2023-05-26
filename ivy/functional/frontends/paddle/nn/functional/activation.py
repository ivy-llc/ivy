# local
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
from ivy.functional.frontends.paddle.tensor.math import relu as paddle_relu
from ivy.functional.frontends.paddle.tensor.math import sigmoid as paddle_sigmoid
from ivy.functional.frontends.paddle.tensor.math import leaky_relu as paddle_leaky_relu


tanh = paddle_tanh
relu=paddle_relu
sigmoid=paddle_sigmoid
leaky_relu=paddle_leaky_relu
