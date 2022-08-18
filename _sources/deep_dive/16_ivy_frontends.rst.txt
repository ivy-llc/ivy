Ivy Frontends
=============

.. _`here`: https://lets-unify.ai/ivy/design/ivy_as_a_transpiler.html
.. _`jax.lax.add`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.add.html
.. _`jax.lax`: https://jax.readthedocs.io/en/latest/jax.lax.html
.. _`jax.lax.tan`: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.tan.html
.. _`numpy.add`: https://numpy.org/doc/stable/reference/generated/numpy.add.html
.. _`numpy mathematical functions`: https://numpy.org/doc/stable/reference/index.html
.. _`numpy.tan`: https://numpy.org/doc/stable/reference/generated/numpy.tan.html
.. _`tf.add`: https://www.tensorflow.org/api_docs/python/tf/math/add
.. _`tf`: https://www.tensorflow.org/api_docs/python/tf
.. _`tf.tan`: https://www.tensorflow.org/api_docs/python/tf/math/tan
.. _`torch.add`: https://pytorch.org/docs/stable/generated/torch.add.html#torch.add
.. _`torch`: https://pytorch.org/docs/stable/torch.html#math-operations
.. _`torch.tan`: https://pytorch.org/docs/stable/generated/torch.tan.html#torch.tan
.. _`YouTube tutorial series`: https://www.youtube.com/watch?v=72kBVJTpzIw&list=PLwNuX3xB_tv-wTpVDMSJr7XW6IP_qZH0t
.. _`ivy frontends discussion`: https://github.com/unifyai/ivy/discussions/2051
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`ivy frontends channel`: https://discord.com/channels/799879767196958751/998782045494976522

Introduction
------------

On top of the Ivy and backend functional APIs, Ivy has another set of
framework-specific frontend functional APIs, which play an important role in code
transpilations, as explained `here`_.

Let's start with some examples to have a better idea on Ivy Frontends!

Basic
-----

**NOTE:** Type hints, docstrings and examples are not required when working on
frontend methods.

**Jax**

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def add(x, y):
        return ivy.add(x, y)

    add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

In the original Jax library, :code:`add` is under `jax.lax.add`_. Thus, an
identical module of :code:`lax` is created and the function is placed there. It
is then categorised under :code:`operators` as shown in the `jax.lax`_ package directory.
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

Looking at a second example, :code:`tan`, it is placed under :code:`operators`
according to the `jax.lax`_ directory. By referring to the `jax.lax.tan`_ documentation,
it has only one argument, and just as our :code:`add` function, we link its return to
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

In NumPy, :code:`add` is categorised under :code:`mathematical_functions` with a
sub-category of :code:`arithmetic_operations` as shown in the
`numpy mathematical functions`_ directory. This ensures direct access to
:code:`numpy.add` in :code:`ivy` by simply importing
:code:`ivy.functional.frontends.numpy`.

The function arguments for this method is slightly more complex due to the extra
optional arguments. Additional handling code is added to recover the behaviour
according to the `numpy.add`_ documentation. For example, if :code:`dtype` is specified,
the arguments to be added will be casted to the desired type through
:code:`ivy.astype`. The returned result is then obtained through :code:`ivy.add`
just like the other examples.

Just like the Jax frontend version of this :code:`add` function, a :code:`dict` is
specified in the :code:`unsupported_dtypes` attribute to indicate the list of
invalid :code:`dtype` for the backend framework.

.. code-block:: python

    # in ivy/functional/frontends/numpy/mathematical_functions/trigonometric_functions.py
    def tan(
        x,
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
            x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
        ret = ivy.tan(x, out=out)
        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
        return ret

    tan.unsupported_dtypes = {"torch": ("float16",)}

With :code:`tan` as the second example, it has a sub-category of
:code:`trigonometric_functions` according to the `numpy mathematical functions`_
directory. By referring to the `numpy.tan`_ documentation, it has additional
arguments just like its :code:`add` function, thus needing additional handling code.

**TensorFlow**

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/functions.py
    def add(x, y, name=None):
        return ivy.add(x, y)

    add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

In the original TensorFlow library (`tf`_ directory), :code:`add` does not have
a specific category. Therefore, it is categorised under :code:`functions` in Ivy.
This ensures that :code:`tf.add` is available directly without further major
changes when using :code:`ivy`. It is valid by simply importing
:code:`ivy.functional.frontends.tensorflow`.

There are three arguments according to the `tf.add`_ documentation, where we
have written accordingly as shown above. Just like the previous examples, it will
also return :code:`ivy.add` for the linking of backend framework. If there are any
unsupported dtypes in any backend, it is specified with the help of the
:code:`unsupported_dtypes` attribute.

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/functions.py
    def tan(x, name=None):
        return ivy.tan(x)

    tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}

Let's look at another example, :code:`tan`, it is placed under :code:`functions` just
like :code:`add`. By referring to the `tf.tan`_ documentation, we code the arguments
accordingly, then link its return to :code:`ivy.tan` so that the computation
operation is decided according to the backend framework.

**PyTorch**

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def add(input, other, *, alpha=1, out=None):
        return ivy.add(input, other * alpha, out=out)

    add.unsupported_dtypes = ("float16",)

For PyTorch, :code:`add` is categorised under :code:`pointwise_ops` as shown in
the `torch`_ directory. This ensures direct access to :code:`torch.add` in :code:`ivy`
without further major changes. It is valid by simply importing
:code:`ivy.functional.frontends.torch`.

For the function arguments, it has to be identical to the original function in
PyTorch to ensure identical behaviour. In this case, the native `torch.add`_ has
both positional and keyword arguments, where we will use the same for our PyTorch
frontend :code:`add`. As for its return, we will link it to :code:`ivy.add` as usual.
However, the arguments work slightly different in this example. From understanding
the PyTorch `torch.add`_ documentation, you will notice that :code:`alpha`
acts as a scale for the :code:`other` argument. Thus, we will recover the original
behaviour by passing :code:`other * alpha` into :code:`ivy.add`.

You may have noticed that the :code:`unsupported_dtypes` attribute is a :code:`tuple`
here. This indicates that this :code:`torch.add` frontend function itself does not
support the :code:`float16` dtype.

.. code-block:: python

    # in ivy/functional/frontends/torch/pointwise_ops.py
    def tan(input, *, out=None):
        return ivy.tan(input, out=out)

    tan.unsupported_dtypes = ("float16",)

Using :code:`tan` as a second example, it is placed under :code:`pointwise_ops`
according to the `torch`_ directory. By referring to the `torch.tan`_ documentation,
we code its positional and keyword arguments accordingly, then return with
:code:`ivy.tan` to link the operation to the backend framework.

Compositions
------------

Sometimes, certain frontend methods meet special criteria:

* it is unique to a particular frontend framework, and does not exist in the other
  frontends, or
* it has extra features than the original backend method

In such cases, compositions are required to recover the original behaviours.

**Examples**

In the native TensorFlow method :code:`tf.cumprod()`, it supports an extra
argument - :code:`reverse`, which returns a flipped result if True. However,
the backend :code:`ivy.cumprod()` does not come with this argument, thus not
supporting this behaviour by default.

**Ivy**

.. code-block:: python

    # in ivy/functional/ivy/general.py
    def cumprod(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        axis: int = 0,
        *,
        exclusive: Optional[bool] = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> Union[ivy.Array, ivy.NativeArray]:
        return current_backend(x).cumprod(x, axis, exclusive, out=out)

To enable this behaviour, we will need to incorporate methods that resemble the
required behaviour. For example, we can reverse the result by calling
:code:`ivy.flip()` after running :code:`ivy.cumprod()`.

**TensorFlow Frontend**

.. code-block:: python

    # ivy/functional/frontends/tensorflow/math/general.py
    def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
        ret = ivy.cumprod(x, axis, exclusive)
        if reverse:
            return ivy.flip(ret, axis)
        return ret

Through compositions, we can easily meet the desired objectives.

Further Examples
----------------

Additional examples for each framework on concatenation is added for better understanding!

**Jax**

.. code-block:: python

    # in ivy/functional/frontends/jax/lax/operators.py
    def concatenate(operands: Sequence[Any], dimension: int) -> Any:
        return ivy.concat(operands, dimension)

**Numpy**

.. code-block:: python

    # in ivy/functional/frontends/numpy/manipulation_routines/joining_arrays.py
    def concatenate(arrays, /, axis=0, out=None, *, dtype=None, casting="same_kind"):
        if dtype:
            arrays = [ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype)) for a in arrays]
        return ivy.concat(arrays, axis, out=out)

**TensorFlow**

.. code-block:: python

    # in ivy/functional/frontends/tensorflow/functions.py
    def concat(values, axis, name="concat"):
        return ivy.concat(values, axis)

**PyTorch**

.. code-block:: python

    # in ivy/functional/frontends/torch/indexing_slicing_joining_mutating_ops.py
    def cat(tensors, dim=0, *, out=None):
        return ivy.concat(tensors, dim, out=out)

**Round Up**

This should hopefully allow you to have a better grasp on the Ivy Frontend APIs
after going through the contents! We have a `YouTube tutorial series`_ on this
as well if you prefer a video explanation!

If you're ever unsure of how best to proceed,
please feel free to engage with the `ivy frontends discussion`_,
or reach out on `discord`_ in the `ivy frontends channel`_!
