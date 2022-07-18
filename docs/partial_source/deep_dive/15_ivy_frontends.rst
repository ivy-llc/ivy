Ivy Frontends
=============

.. _`jax.lax.add`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.add.html

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

In the original Jax library, :code:`add` is under `jax.lax.add`_

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

**PyTorch**

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def add(input, other, *, alpha=1, out=None):
        return ivy.add(input, other * alpha, out=out)

    add.unsupported_dtypes = ("float16",)

**TensorFlow**

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/functions.py
    def add(x, y, name=None):
        return ivy.add(x, y)

    add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
