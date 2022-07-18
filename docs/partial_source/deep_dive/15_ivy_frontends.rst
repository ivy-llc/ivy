Ivy Frontends
=============

.. _`jax.lax.add`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.add.html
.. _`jax.lax`: https://jax.readthedocs.io/en/latest/jax.lax.html
.. _`jax.lax.tan`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.tan.html

Introduction
------------

On top of the Ivy and backends functional APIs, Ivy has another set of
framework-specific frontend functional APIs, which play an important role in the
Ivy transpilation feature. With this, Ivy's graph compiler will be able to analyse
user's code, then produce an equivalent computation graph where operations are
specific to the framework set.

The frontend functional APIs will also be coded according to the module structure
in their respective framework. This is to ensure that codes are framework agnostic.
For example, a certain piece of completed PyTorch code can be directly used with
Ivy without changes due to the presence of this set of APIs.

Let's start with some examples to have a better idea on Ivy Frontends!

Examples
--------

**Jax**

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def add(x, y):
        return ivy.add(x, y)

    add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

In the original Jax library, :code:`add` is under `jax.lax.add`_. Thus, an
identical module of :code:`lax` is created and the function is placed there. It
is then categorised under operators as shown in the `jax.lax`_ package directory.
This is to ensure that :code:`jax.lax.add` is available directly without further
major changes when using :code:`ivy`. It is valid by simply importing
:code:`ivy.functional.frontends.jax`.

For the function arguments, it has to be identical to the original function in
Jax to ensure identical behaviour. In this case, `jax.lax.add`_ has two arguments,
where we will also have the same two arguments in our Jax frontend :code:`add`.
Then, this function will return :code:`ivy.add`, which links to the :code:`add`
function according to the framework set in the backend.

You may have noticed that the function has an additional attribute
:code:`unsupported_dtypes`. This is because users are allowed to set a backend
operating framework which is not Jax. There may be certain :code:`dtype` which
the backend cannot support, for instance, PyTorch does not support
:code:`float16` and :code:`bfloat16` in their :code:`add` function. These are then
specified with the help of this attribute.

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def tan(x):
        return ivy.tan(x)

    tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

Looking at a second example, :code:`tan`, it is placed under operators according to
the `jax.lax`_ directory. By referring to the `jax.lax.tan`_ documentation, it
has only one argument, and just as our :code:`add` function, we link its return to
:code:`ivy.tan` so that the computation operation depends on the backend framework.

**NumPy**

.. code-block:: python

    # in ivy/functional/frontends/numpy/mathematical_functions/arithmetic_operations.py
    def add(
        x1,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="k",
        dtype=None,
        subok=True,
    ):
        if dtype:
            x1 = ivy.astype(ivy.array(x1), ivy.as_ivy_dtype(dtype))
            x2 = ivy.astype(ivy.array(x2), ivy.as_ivy_dtype(dtype))
        ret = ivy.add(x1, x2, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

    add.unsupported_dtypes = {"torch": ("float16",)}

**TensorFlow**

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/functions.py
    def add(x, y, name=None):
        return ivy.add(x, y)

    add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

**PyTorch**

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def add(input, other, *, alpha=1, out=None):
        return ivy.add(input, other * alpha, out=out)

    add.unsupported_dtypes = ("float16",)
