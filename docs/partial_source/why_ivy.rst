Why Ivy?
========

Portable Code
-------------

Ivy's strength arises when we want to maximize the usability of our code.

With machine learning and gradient-based optimization increasingly finding their way into all kinds of different fields,
let's suppose you need to implement a set of functions in one of these fields for your own DL project.

The topic could be anything, such as bayesian inference, medical imaging, fluid mechanics, particle physcis, economics etc.
For the purpose of this example, let's assume you need to implement some functions for bayesian inference.

First, let's consider you write your functions directly in pytorch.
After several days, you then finish implementing all functions, and decide you would like to open source them.
One of these functions is given below, for a single step of a kalman filter:

.. code-block:: python

    def kalman_filter_step(xkm1km1, Pkm1km1, uk, Fk, Bk, Qk, Hk, Rk):
        # reference: https://en.wikipedia.org/wiki/Kalman_filter#Details

        # trans
        FkT = torch.swapaxes(Fk, (-1, -2))
        HkT = torch.swapaxes(Hk, (-1, -2))

        # predict
        xkkm1 = torch.matmul(Fk, xkm1km1) + torch.matmul(Bk, uk)
        Pkkm1 = torch.matmul(torch.matmul(Fk, Pkm1km1), FkT) + Qk

        # update
        Sk = torch.matmul(torch.matmul(Hk, Pkkm1), HkT) + Rk
        Skinv = torch.inverse(Sk)
        Kk = torch.matmul(torch.matmul(Pkkm1, HkT), Skinv)
        ImKkHk = I - torch.matmul(Kk, Hk)
        xkk = torch.matmul(ImKkHk, xkkm1)  + torch.matmul(Kk, torch.matmul(Hk, xk) + vk)
        Pkk = torch.matmul(ImKkHk, Pkkm1)

        # return
        return xkk, Pkk

Naturally, you would like your code to get as much recognition as possible, with a maximal number of users.
But with every function in your library written in pure pytorch, this closes the door on TensorFlow, Jax, and MXNet users.

In this simple case, manual reimplementation would be feasible,
but for more complex libraries and codebases this becomes a significant time investment.

Furthermore, the most popular machine learning framework in 2 years time may not even exist yet.
Your pytorch library would then inevitably become outdated.

Ivy solves this combination of problems. Firstly, if you had instead written your library in Ivy,
your library would immediately be usable for developers in all current machine learning frameworks.
Secondly, the Ivy team are commited to keeping Ivy compatible with any new frameworks introduced in future,
meaning your library will not become outdated.

Writing code in Ivy is just as easy as writing code in any other framework,
we show the same kalman filter function written in Ivy below.

.. code-block:: python

    import ivy

    def kalman_filter_update(xkm1km1, Pkm1km1, zk, Rk, uk, Fk, Bk, Qk, Hk):
        # reference: https://en.wikipedia.org/wiki/Kalman_filter#Details

        # trans
        FkT = ivy.swapaxes(Fk, (-1, -2))
        HkT = ivy.swapaxes(Hk, (-1, -2))

        # predict
        xkkm1 = ivy.matmul(Fk, xkm1km1) + ivy.matmul(Bk, uk)
        Pkkm1 = ivy.matmul(ivy.matmul(Fk, Pkm1km1), FkT) + Qk

        # update
        Sk = ivy.matmul(ivy.matmul(Hk, Pkkm1), HkT) + Rk
        Skinv = ivy.inv(Sk)
        Kk = ivy.matmul(ivy.matmul(Pkkm1, HkT), Skinv)
        ImKkHk = I - ivy.matmul(Kk, Hk)
        xkk = ivy.matmul(ImKkHk, xkkm1)  + ivy.matmul(Kk, ivy.matmul(Hk, xk) + vk)
        Pkk = ivy.matmul(ImKkHk, Pkkm1)

        # return
        return xkk, Pkk

The backend framework can be selected before calling the function like so ``ivy.set_framework('torch')``.
Further details on how to write efficient Ivy code are given in the short guide `Using Ivy <https://lets-unify.ai/ivy/using_ivy.html>`_.

We now consider the use of your new library by some hypothetical users,
with this particular application inspired by `Backprop Kalman Filter <https://arxiv.org/abs/1605.07148>`_.
The details of the algorithm are not important,
we simply aim to highlight the "drag-and-drop" nature of this new bayesian Ivy library.

For a tensorflow developer using your library, their network class might look something like this:

.. code-block:: python

    # Tensorflow User

    import ivy
    import ivy_bayes
    import tensorflow as tf
    ivy.set_framework('tensorflow')

    class Network(tf.keras.layers.Layer):
        def __init__(self):
            self._unroll_steps = 10
            self._model = _get_some_model()

        def call(self, zk, Rk)
            mean, variance = self._get_prior()
            for _ in range(self._unroll_steps):
                zk_e, Rk_e = self._model(zk, Rk)
                mean, variance = ivy_bayes.kalman_filter_update(
                mean, var, zk_e, Rk_e, *self._get_kalman_params())

For a pytorch developer using your library, their network class might look something like this:

.. code-block:: python

    # PyTorch User

    import torch
    import ivy_bayes
    import ivy
    ivy.set_framework('torch')

    class Network(torch.nn.Module):
        def __init__(self):
            self._unroll_steps = 10
            self._model = _get_some_model()

        def call(self, zk, Rk)
            mean, variance = self._get_prior()
            for _ in range(self._unroll_steps):
                zk_e, Rk_e = self._model(zk, Rk)
                mean, variance = ivy_bayes.kalman_filter_update(
                mean, var, zk_e, Rk_e, *self._get_kalman_params())

The same drag-and-drop behaviour is possible for MXNet, Jax and Numpy,
and we are commited to supporting future machine learning frameworks, yet to be created.
