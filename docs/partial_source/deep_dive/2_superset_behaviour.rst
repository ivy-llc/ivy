Superset Behaviour
==================

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`superset behavior channel`: https://discord.com/channels/799879767196958751/1018954266322419732
.. _`superset behavior discussion`: https://github.com/unifyai/ivy/discussions/4367

When implementing functions in Ivy, we are constantly faced with the question: which
backend implementation should Ivy most closely follow?

It might seem as though this question is already answered.
Ivy fully adheres to the `Array API Standard`_, which helpfully limits our design space
for the functions, but in its current form this only covers a relatively small number of
functions which together make up less than half of the functions in Ivy.
Even for Ivy functions which adhere to the standard,
the standard permits the addition of extra arguments and function features,
provided that they do not contradict the requirements of the standard.
Therefore, we are still faced with the same kind of design decisions for all Ivy
functions, even those appearing in the `Array API Standard`_.

In this section, we explain through examples how Ivy always goes for the superset of
functionality among the backend frameworks. This is not always totally possible,
and in some cases certain framework-specific features must be sacrificed,
but usually it's possible to implement a very generalized function which covers most of
the unique features among the corresponding functions in each framework.

Despite this general approach, the total superset is not always strived for, especially
in cases where the behaviour can very easily be replicated by a simple composition of
other functions, or where the extra behaviour is redundant as it is already covered by
another function.
As an example, many pointwise functions in NumPy support the :code:`where` argument,
which enables a mask array to be specified, with the function then only evaluated
at elements for which the :code:`where` array is :code:`True`.
This inclusion of this feature in NumPy is totally understandable,
compositions of NumPy functions are never compiled into computation graph which span
multiple operations, and therefore a good way to maximize efficiency of NumPy code is to
minimize the number of unique NumPy functions which are called, each of which are
implemented with very high efficiency in :code:`C`. In this case, the inclusion of
:code:`where` as an argument also prevents unnecessary values from being computed,
only to be masked out in the immediately subsequent operation.
For these reasons, calling :code:`np.absolute(x, where=mask)` is much better than
calling :code:`np.where(mask, np.absolute(x), np.empty_like(x))` in NumPy.
However, other frameworks are able to compile compositions of python operations directly
to computation graphs in low-level languages, and are also able to intelligently fuse
operations into combined kernels, via libraries such as
`TensorRT <https://github.com/NVIDIA/TensorRT>`_.
This removes the need for highly general function signatures such as those found in
NumPy. Instead, a compositional approach is preferred, where each function in Python
serves a particular and non-overlapping purpose.
This helps to keep things more clean and clear at the Python level,
without sacrificing efficiency at the lower level.

We now take a look at some examples, and explain our rational for deciding upon the
function signature that we should use in Ivy. The first three examples are more-or-less
superset examples, while the last example involves a deliberate decision to not
implement the full superset, for the reason explained above.


ivy.linspace
------------

When looking at the :code:`linspace` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/creation/linspace/linspace_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linspace.html>`_,
`NumPy <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/linspace>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.linspace.html>`_,
we can see that :code:`torch` does not support arrays for the :code:`start` and
:code:`end` arguments, while :code:`jax`, :code:`numpy` and :code:`tensorflow` all do.
Likewise, Ivy also supports arrays for the :code:`start` and :code:`stop` arguments,
and in doing so provides the generalized superset implementation among the backend
frameworks.


ivy.eye
-------

When looking at the :code:`eye` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/creation/eye/eye_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.eye.html>`_,
`NumPy <https://numpy.org/devdocs/reference/generated/numpy.eye.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/eye>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.eye.html>`_,
we can see that :code:`tensorflow` is the only framework which supports a
:code:`batch_shape` argument. Likewise, Ivy also supports a :code:`batch_shape`
argument, and in doing so provides the generalized superset implementation among the
backend frameworks.


ivy.scatter_nd
--------------

When looking at the :code:`scatter_nd` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/general/scatter_nd/scatter_nd_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at>`_,
`NumPy <https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/scatter_nd>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.scatter.html>`_,
we can see that :code:`torch` only supports scattering along a single dimension,
while all other frameworks support scattering across multiple dimensions at once.
Likewise, Ivy also supports scattering across multiple dimensions at once,
and in doing so provides the generalized superset implementation among the backend
frameworks.


ivy.logical_and
---------------

When looking at the :code:`logical_and` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/elementwise/logical_and/logical_and_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logical_and.html>`_,
`NumPy <https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/math/logical_and>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.logical_and.html>`_,
we can see that :code:`numpy` and :code:`torch` support the :code:`out` argument for
performing inplace updates, while :code:`jax` and :code:`tensorflow` do not.
With regards to the supported data types, :code:`jax`, :code:`numpy` and :code:`torch`
support numeric arrays, while :code:`tensorflow` supports only boolean arrays.
With regards to both of these points, Ivy provides the generalized superset
implementation among the backend frameworks, with support for the :code:`out` argument
and also support for both numeric and boolean arrays in the input.

However, as discussed above, :code:`np.logical_and` also supports the :code:`where`
argument, which we opt to **not** support in Ivy. This is because the behaviour can
easily be created as a composition like so
:code:`ivy.where(mask, ivy.logical_and(x, y), ivy.zeros_like(mask))`,
and we prioritize the simplicity, clarity, and function uniqueness in Ivy's API in this
case, which comes at the cost of reduced runtime efficiency for some functions when
using a NumPy backend. However, in future releases our automatic graph compilation and
graph simplification processes will alleviate these minor inefficiencies entirely from
the final computation graph, by fusing multiple operations into one at the API level
where possible.


**Round Up**

This should have hopefully given you a good feel what should and should not be included
when deciding how to design a new Ivy function.
In many cases, there is not a clear right and wrong answer, and we arrive at the final
decision via open discussion. If you find yourself proposing the addition of a new
function in Ivy, then we will most likely have this discussion on your Pull Request!

If you're ever unsure of how best to proceed,
please feel free to engage with the `superset behavior discussion`_,
or reach out on `discord`_ in the `superset behavior channel`_!
