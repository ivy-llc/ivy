Ivy Stateful API
================

Here we explain how Ivy’s stateful API builds on the functional API and the :class:`ivy.Container` class to provide other convenient classes in the form of optimizers, network layers and custom trainable modules, which help get your ML projects up and running very quickly!

So, without further ado, let’s walk through what the stateful API has to offer!

Modules
-------

The most helpful stateful Ivy class is perhaps the :class:`ivy.Module`.
This can be used to create custom trainable layers or entire networks.
Manually defined trainable variables must be specified in the :meth:`_create_variables` method.
For example, we can create a linear layer by deriving from :class:`ivy.Module` like so:

.. code-block:: python

    class Linear(ivy.Module):

        def __init__(self, input_channels, output_channels,
                     with_bias=True, dev=None, v=None):
            self._input_channels = input_channels
            self._output_channels = output_channels
            self._w_shape = (output_channels, input_channels)
            self._b_shape = (output_channels,)
            self._with_bias = with_bias
            ivy.Module.__init__(self, dev, v)

        def _create_variables(self, dev):
            v = {'w': ivy.random_uniform(
                    shape=self._w_shape, dev=dev)}
            if self._with_bias:
                v = dict(**v, b=ivy.random_uniform(
                            shape=self._b_shape, dev=dev))
            return v

        def _forward(self, inputs):
            return ivy.linear(
              inputs, self.v.w,
              self.v.b if self._with_bias else None)

For simplicity, this is slightly different to the builtin :class:`ivy.Linear` in a couple of ways, as we will explain in the Initializer section below.

All :class:`ivy.Module` instances have an attribute v (short for variables), which stores all of the trainable variables in the module in an :class:`ivy.Container`.
For our example above, the hierarchical structure of these variables is the same as that defined in the method :meth:`_create_variables`.

.. code-block:: python

    linear = Linear(2, 4)
    print(linear.v)

    {
        b: ivy.array([0., 0., 0., 0.]),
        w: ivy.array([[-0.729, 0.396],
                      [-1., -0.764],
                      [-0.872, 0.211],
                      [0.439, -0.644]])
    }

This is all well and good for defining a single layer, but manually defining all variables in :code:`_create_variables` for very complex networks would be a total nightmare.

To overcome this issue, modules can be nested up to an arbitrary depth.
This means we can very easily create more complex networks as compositions of other sub-modules or layers.
For example, we can create a simple fully connected network with our linear layers.

.. code-block:: python

    class FC(ivy.Module):
        def __init__(self):
            self.linear0 = Linear(3, 64)
            self.linear1 = Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))

In this case, we don’t specify any variables manually using :code:`_create_variables`.
This is because all variables in the network reside in the linear layers.
These variables are all detected automatically.

.. code-block:: python

    fc = FC()
    print(fc.v)

    {
        linear0: {
            b: (<class ivy.array.array.Array> shape=[64]),
            w: (<class ivy.array.array.Array> shape=[64, 3])
        },
        linear1: {
            b: ivy.array([0.]),
            w: (<class ivy.array.array.Array> shape=[1, 64])
        }
    }

Not only are variables detected automatically for :class:`ivy.Module` instances which are direct attributes of the top-level class, as above, but also if they are contained within any nested structure which is itself an attribute of the top-level class, such as lists, tuples or dicts.
These all work up to an arbitrary nested depth.
Check out some of the different ways of defining network layers and how this impacts the variable structure below.

As a list:

.. code-block:: python

    class FC(ivy.Module):
        def __init__(self):
            self.linear = [Linear(3, 64), Linear(64, 1)]
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear[0](x))
            return ivy.sigmoid(self.linear[1](x))

    fc = FC()
    print(fc.v)

    {
        linear: {
            v0: {
                b: (<class ivy.array.array.Array> shape=[64]),
                w: (<class ivy.array.array.Array> shape=[64, 3])
            },
            v1: {
                b: ivy.array([0.]),
                w: (<class ivy.array.array.Array> shape=[1, 64])
            }
        }
    }

As a tuple:

.. code-block:: python

    class FC(ivy.Module):
        def __init__(self):
            self.linear = (Linear(3, 64), Linear(64, 1))
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear[0](x))
            return ivy.sigmoid(self.linear[1](x))

    fc = FC()
    print(fc.v)

    {
        linear: {
            v0: {
                b: (<class ivy.array.array.Array> shape=[64]),
                w: (<class ivy.array.array.Array> shape=[64, 3])
            },
            v1: {
                b: ivy.array([0.]),
                w: (<class ivy.array.array.Array> shape=[1, 64])
            }
        }
    }

As a dict:

.. code-block:: python

    class FC(ivy.Module):
        def __init__(self):
            self.linear = {'key0': Linear(3, 64),
                           'key1': Linear(64, 1)}
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear['key0'](x))
            return ivy.sigmoid(self.linear['key1'](x))

    fc = FC()
    print(fc.v)

    {
        linear: {
            key0: {
                b: (<class ivy.array.array.Array> shape=[64]),
                w: (<class ivy.array.array.Array> shape=[64, 3])
            },
            key1: {
                b: ivy.array([0.]),
                w: (<class ivy.array.array.Array> shape=[1, 64])
            }
        }
    }

As a nested list:

.. code-block:: python

    class FC(ivy.Module):
        def __init__(self):
            self.linear = [[Linear(3, 64), Linear(64, 64)],
                           Linear(64, 1)]
            ivy.Module.__init__(self)

        def _forward(self, x):
            for linear in self.linear[0]:
                x = ivy.relu(linear(x))
            return ivy.sigmoid(self.linear[1](x))

    fc = FC()
    print(fc.v)

    {
        linear: {
            v0: {
                v0: {
                    b: (<class ivy.array.array.Array> shape=[64]),
                    w: (<class ivy.array.array.Array> shape=[64, 3])
                },
                v1: {
                    b: (<class ivy.array.array.Array> shape=[64]),
                    w: (<class ivy.array.array.Array> shape=[64, 64])
                }
            },
            v1: {
                b: ivy.array([0.]),
                w: (<class ivy.array.array.Array> shape=[1, 64])
            }
        }
    }

Duplicates are also handled correctly, if for example a layer is stored both as a direct attribute and also within a list:

.. code-block:: python

    class FC(ivy.Module):
        def __init__(self):
            self.linear0 = Linear(3, 64)
            self.linear1 = Linear(64, 64)
            self.linear3 = Linear(64, 1)
            self.linear = [self.linear0,
                           self.linear1,
                           Linear(64, 64)]
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear[0](x))
            x = ivy.relu(self.linear[1](x))
            x = ivy.relu(self.linear[2](x))
            return ivy.sigmoid(self.linear3(x))

    fc = FC()
    print(fc.v)

    {
        linear: {
            v0: {
                b: (<class ivy.array.array.Array> shape=[64]),
                w: (<class ivy.array.array.Array> shape=[64, 3])
            },
            v1: {
                b: (<class ivy.array.array.Array> shape=[64]),
                w: (<class ivy.array.array.Array> shape=[64, 64])
            },
            v2: {
                b: (<class ivy.array.array.Array> shape=[64]),
                w: (<class ivy.array.array.Array> shape=[64, 64])
            }
        },
        linear3: {
            b: ivy.array([0.]),
            w: (<class ivy.array.array.Array> shape=[1, 64])
        }
    }

While the examples above all use the functional API for calling the ReLU and Sigmoid activation functions, we can also call these using the stateful API like so:

.. code-block:: python

    class FC(ivy.Module):
        def __init__(self):
            self.linear0 = Linear(3, 64)
            self.linear1 = Linear(64, 1)
            self.relu = ivy.ReLU()
            self.sigmoid = ivy.Sigmoid()
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = self.relu(self.linear0(x))
            return self.sigmoid(self.linear1(x))

It may seem counter intuitive to implement the activation as an :class:`ivy.Module`, as there are no hidden trainable weights.
However, for networks where modules are directly chained together, and all outputs from the preceding module are fed as inputs to the subsequent module, then we can use the :class:`ivy.Sequential` class.
This can simplify the construction of our small fully connected network even further.

.. code-block:: python

    fc = ivy.Sequential(
            Linear(3, 64),
            ivy.ReLU(),
            Linear(64, 1),
            ivy.Sigmoid())

    print(fc.v)

    {
        submodules: {
            v0: {
                b: (<class ivy.array.array.Array> shape=[64]),
                w: (<class ivy.array.array.Array> shape=[64, 3])
            },
            v2: {
                b: ivy.array([0.]),
                w: (<class ivy.array.array.Array> shape=[1, 64])
            }
        }
    }

Given that the weights of our network are stored in an :class:`ivy.Container`, and the gradients returned from :func:`ivy.execute_with_gradients` are also stored in an :class:`ivy.Container`, all operations are applied recursively to every variable at all leaves.
Therefore, we can train the network in a few lines of code like so:

.. code-block:: python

    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])
    lr = 0.001

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.reduce_mean((out - target)**2)[0]

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(
            loss_fn, model.v)
        model.v = model.v - grads * lr

Initializers
------------

In the examples above, we defined how the trainable weights should be initialized directly in the :code:`_create_variables` method.
However, it would be better if we could decouple the initialization scheme from the layer implementation.
This is where the :class:`ivy.Initializer` class comes in.
The actual implementation for the :class:`ivy.Linear` layer exposed in the Ivy stateful API is as follows:

.. code-block:: python

    # ivy/stateful/layers.py
    class Linear(ivy.Module):

        def __init__(self, input_channels, output_channels,
                     weight_initializer=GlorotUniform(),
                     bias_initializer=Zeros(), with_bias=True,
                     dev=None, v=None):
            self._input_channels = input_channels
            self._output_channels = output_channels
            self._w_shape = (output_channels, input_channels)
            self._b_shape = (output_channels,)
            self._w_init = weight_initializer
            self._b_init = bias_initializer
            self._with_bias = with_bias
            ivy.Module.__init__(self, dev, v)

        def _create_variables(self, dev):
            v = {'w': self._w_init.create_variables(
              self._w_shape, dev, self._output_channels,
              self._input_channels)}
            if self._with_bias:
                v = dict(**v, b=self._b_init.create_variables(
                  self._b_shape, dev, self._output_channels))
            return v

        def _forward(self, inputs):
            return ivy.linear(
              inputs, self.v.w,
              self.v.b if self._with_bias else None)

The :class:`ivy.Initializer` class has a single abstract method, :code:`create_variables(var_shape, dev, fan_out=None, fan_in=None, *args, **kwargs)`.
Check out the `code <https://github.com/unifyai/ivy/blob/master/ivy/stateful/initializers.py>`_ or `docs <https://unify.ai/docs/ivy/neural_net_stateful/initializers.html>`_ for more details.
The default initializer for the weights is :class:`ivy.GlorotUniform` and for this bias is :class:`ivy.Zeros`.
Let’s take a quick look at what these look like.
:class:`ivy.GlorotUniform` derives from a more general :class:`ivy.Uniform` initializer class, and is then simply implemented as follows:

.. code-block:: python

    # ivy/stateful/initializers.py
    class GlorotUniform(ivy.Uniform):
        def __init__(self):
            super().__init__(
                numerator=6, fan_mode='fan_sum', power=0.5, gain=1)

:class:`ivy.Zeros` derives from a more general :class:`ivy.Constant` initializer class, and is then simply implemented as follows:

.. code-block:: python

    # ivy/stateful/initializers.py
    class Zeros(ivy.Constant):
        def __init__(self):
            super().__init__(constant=0.)

The initializers are not stateful, and so adding them to the “stateful API” is a slight misnomer.
However, the dedicated initializer class helps us to decouple initialization schemes from layer implementations, which are themselves stateful.
Given that their application is entirely specific to stateful :class:`ivy.Module` instances, they still belong in the stateful API.

Optimizers
----------
Recapping the example given above, we saw that :class:`ivy.Module` instances can be trained like so:

.. code-block:: python

    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])
    lr = 0.001

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.reduce_mean((out - target)**2)[0]

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(
            loss_fn, model.v)
        model.v = model.v - grads * lr

However, what if we want to do something more complex than vanilla gradient descent? What about ADAM or other stateful optimizers such as LARS and LAMB? This is where the :class:`ivy.Optimizer` class comes in.

Let’s take the class :class:`ivy.Adam` as an example.
The implementation is as follows:

.. code-block:: python

    # ivy/stateful/optimizers.py
    class Adam(ivy.Optimizer):

        def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999,
                     epsilon=1e-07, inplace=None,
                     stop_gradients=True, compile_on_next_step=False,
                     dev=None):
            ivy.Optimizer.__init__(
                self, lr, inplace, stop_gradients, True,
                compile_on_next_step, dev)
            self._beta1 = beta1
            self._beta2 = beta2
            self._epsilon = epsilon
            self._mw = None
            self._vw = None
            self._first_pass = True
            self._should_compile = False

        # Custom Step

        def _step(self, v, grads):
            if self._first_pass:
                self._mw = grads
                self._vw = grads ** 2
                self._first_pass = False
            new_v, self._mw, self._vw = ivy.adam_update(
                v, grads, self._lr if isinstance(self._lr, float)
                else self._lr(), self._mw, self._vw, self._count,
                self._beta1, self._beta2, self._epsilon, self._inplace,
                self._stop_gradients)
            return new_v

        def set_state(self, state):
            self._mw = state.mw
            self._vw = state.vw

        @property
        def state(self):
            return ivy.Container({'mw': self._mw, 'vw': self._vw})

By changing only a couple of lines, we can use this optimizer to train our network like so:

.. code-block:: python

    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])
    optimizer = ivy.Adam(0.001)

    def loss_fn(v):
        out = model(x_in, v=v)
        return ivy.reduce_mean((out - target)**2)[0]

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(
            loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)

**Round Up**

That should hopefully be enough to get you started with Ivy’s stateful API 😊

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!
