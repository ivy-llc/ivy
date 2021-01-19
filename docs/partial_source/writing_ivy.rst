Writing Ivy
===========

Ivy Namespace
-------------

Writing Ivy code is simple.
The :code:`ivy` namespace can be used directly with tensors from any of the supported deep learning frameworks.
While this is the simplest way of using Ivy, it is also the least efficient.
In this format, Ivy will type-check the inputs on each function call.

.. code-block:: python

    # Using the ivy namespace, with internal type-checking of inputs

    import tensorflow as tf

    import ivy

    tf_concatted = ivy.concatenate((tf.ones((1,)), tf.ones((1,))), -1)



Ivy makes use of a function :code:`get_framework` for doing this internally,
which performs string parsing to determine the input data types.
This method can also be called outside of ivy functions:

.. code-block:: python

    # using get_framework directly, which type-checks the inputs

    import tensorflow as tf
    import torch

    from ivy.framework_handler import get_framework

    # returns ivy.tensorflow
    get_framework(tf.constant([1.]))

    # returns ivy.torch
    get_framework(torch.tensor([1.]))

Using the :code:`ivy` namespace directly is okay for prototyping, or if runtime is not a high priority.
But if runtime is important, which is certainly the case in most deep learning applications,
then using the :code:`ivy` namespace directly is not advised.

Ivy Frameworks
--------------

In addition to the core :code:`ivy` namespace, Ivy also exposes frameworks, in the form of :code:`ivy.framework`.
These are actually what are returned from the :code:`get_framework` method, as in the examples above.

For writing more efficient code, these Ivy framework modules should be imported and used directly.
These Ivy frameworks bind directly to the functions of the supported backend frameworks.

.. code-block:: python

    # Using Ivy frameworks directly

    import tensorflow as tf

    import ivy.tensorflow as ivy_tf

    # no type-checking, uses direct bindings to tensorflow functions
    tf_concatted = ivy_tf.concatenate((tf.ones((1,)), tf.ones((1,))), -1)


This may seem to entirely defeat the purpose of Ivy, given that the imported :code:`ivy_tf` module is now framework-specific,
why not just use :code:`tf.concat` directly in the example above?

Importantly, :code:`ivy.tensorflow` still maintains the :code:`ivy` syntax and call signatures,
which are consistent across all backend frameworks,
and therefore are distinct from the syntax and call signatures of pure tensorflow, imported as :code:`tf` above.

Ivy placeholders
----------------

In order to make use of the efficient Ivy frameworks, and their direct function bindings,
placeholders can be used in newly written Ivy functions, such as :code:`f` in the example below.

.. code-block:: python

    # Using Ivy placeholders

    import tensorflow as tf
    import torch

    import ivy.tensorflow as ivy_tf
    import ivy.torch as ivy_torch

    def sin_cos(x, f):
        return f.cos(f.sin(x))

    # no type-checking
    sin_cos(tf.constant([1.]), ivy_tf)

    # no type-checking
    sin_cos(torch.tensor([1.]), ivy_torch)

If you would like a function to keep input type-checking as an option, the :code:`get_framework` method can be used.
If :code:`f` is passed into :code:`get_framework`, then type-checking is entirely bypassed,
and :code:`get_framework` immediately returns the input framework :code:`f`.

.. code-block:: python

    # Using Ivy placeholders with optional type-checking

    import tensorflow as tf
    import torch

    import ivy.tensorflow as ivy_tf
    import ivy.torch as ivy_torch
    from ivy.framework_handler import get_framework

    def sin_cos(x, f=None):
        f = get_framework(x, f=f)
        return f.cos(f.sin(x))

    # type-checking
    sin_cos(tf.constant([1.]))

    # no type-checking
    sin_cos(tf.constant([1.]), ivy_tf)

    # type-checking
    sin_cos(torch.array([1.]))

    # no type-checking
    sin_cos(torch.array([1.]), ivy_torch)


The use of Ivy placeholders is also useful for writing Ivy classes,
where the framework can be stored as part of the object state.

.. code-block:: python

    # Using Ivy placeholders with classes

    import tensorflow as tf

    import ivy.tensorflow as ivy_tf
    from ivy.framework_handler import get_framework

    class MyClass:

        def __init__(self, config, f=None):
            self._config = config
            self._f = get_framework(config, f=f)

        def some_fn(x)
            return self._f.cos(self._config * x)

    # type-checking
    obj = MyClass(tf.constant([0.]))

    # no type-checking
    obj.some_fn(tf.constant([1.]))