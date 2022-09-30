Superset Behaviour
==================

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`superset behavior channel`: https://discord.com/channels/799879767196958751/1018954266322419732
.. _`superset behavior discussion`: https://github.com/unifyai/ivy/discussions/4367

When implementing functions in Ivy, whether they are primary, compositional or mixed,
we are constantly faced with the question: which backend implementation should Ivy most
closely follow?

Extending the Standard
----------------------

It might seem as though this question is already answered.
Ivy fully adheres to the `Array API Standard`_, which helpfully limits our design space
for the functions, but in its current form this only covers a relatively small number of
functions, which together make up less than half of the functions in Ivy.
Even for Ivy functions which adhere to the standard,
the standard permits the addition of extra arguments and function features,
provided that they do not contradict the requirements of the standard.
Therefore, we are still faced with the same kind of design decisions for all Ivy
functions, even those appearing in the `Array API Standard`_.

What is the Superset?
---------------------

We explain through examples how Ivy always goes for the superset of
functionality among the backend frameworks. This means that even if only one framework
supports a certain feature, then we still strive to include this feature in the Ivy
function. The Ivy function then entails the *superset* of all backend features.
However, this is not always totally possible, and in some cases certain
framework-specific features must be sacrificed, but usually it's possible to implement a
very generalized function which covers most of the unique features among the
corresponding functions in each framework.

We strive to implement the superset for primary, compositional and mixed functions. In
many cases compositional functions do not actually have corresponding backend-specific
functions, but this is not always the case. For example, :func:`ivy.linear` is a
fully compositional function, but :func:`torch.nn.functional.linear` also exists.
We should therefore make sure the compositional :func:`ivy.linear` function includes all
behaviours supported by :func:`torch.nn.functional.linear`.

A Non-Duplicate Superset
------------------------

It would be easy to assume that implementing the superset simply means adding all
arguments from all related functions into the Ivy function. However, this is **not** the
case for a few reasons. Firstly, different functions might have different argument names
for the same behaviour. Looking at the functions
`numpy.concatenate <https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html>`_
and
`torch.cat <https://pytorch.org/docs/stable/generated/torch.cat.html>`_,
we of course do not want to add both of the arguments :code:`axis` and :code:`dim` to
:func:`ivy.concat`, as these both represent exactly the same thing: the dimemsion/axis
along which to concatenate. In this case, the argument is
`covered <https://data-apis.org/array-api/latest/API_specification/generated/signatures.manipulation_functions.concat.html>`_
in the `Array API Standard`_ and so we opt for :code:`axis`. In cases where there are
differences between the backend argument names, and the function or argument is not in
the standard, then it is up to us to determine which argument name to use.

What is not the Superset?
-------------------------

We've already explained that we should not duplicate arguments in the Ivy function when
striving for the superset. Does this mean, provided that the proposed argument is not a
duplicate, that we should always add this backend-specific argument to the Ivy function?
The answer is **no**. When determining the superset, we are only concerned with the
pure **mathematics** of the function, and nothing else. For example, the :code:`name`
argument is common to many TensorFlow functions, such as
`tf.concat <https://www.tensorflow.org/api_docs/python/tf/concat>`_,
and is used for uniquely identifying parts of the compiled computation graph during
logging and debugging. This has nothing to do with the mathematics of the function, and
so is *not* included in the superset considerations when implementing Ivy functions.
Similarly, in NumPy the argument :code:`subok` controls whether subclasses of the
:code:`numpy.ndarray` class should be permitted, and :code:`order` controls the
low-level memory layout of the array, both of which are included for many functions,
such as `numpy.ndarray.astype <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html>`_.
Finally, in JAX the argument :code:`precision` is quite common, which controls the
precision of the return values, as used in
`jax.lax.conv <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv.html>`_
for example. Similarly, the functions :code:`jacfwd` and :code:`jacrev` in JAX are
actually mathematically identical, and these functions differ *only* in their underlying
algorithm, either forward mode or reverse mode.

None of the above arguments or function variants are included in our superset
considerations, as again they are not relating to the pure mathematics, and instead
relate to framework, hardware or algorithmic specifics. Given the abstraction layer that
Ivy operates at, Ivy is fundamentally unable to control under-the-hood specifics such as
those mentioned above. However, this is by design, and the central benefit of Ivy is the
ability to abstract many different runtimes and algorithms under the same banner,
unified by their shared fundamental mathematics.

Regarding the **only mathematics** rule regarding the superset considerations, there are
two exceptions to this, which are the handling of data type and device arguments.
Neither of these relate to the pure mathematics of the function. However, as is
discussed below, we always strive to implement Ivy functions such that they support as
many data types and devices as possible.

When the Superset is Too Much
-----------------------------

Despite this general approach, the total superset is not always actually strived for,
especially in cases where the behaviour can very easily be replicated by a simple
composition of other functions, or where the extra behaviour is redundant as it is
already covered by another function.

As an example, many pointwise functions in NumPy support the :code:`where` argument,
which enables a mask array to be specified, with the function then only evaluated
at elements for which the :code:`where` array is :code:`True`.
This inclusion of this feature in NumPy is totally understandable,
compositions of NumPy functions are never compiled into computation graphs which span
multiple operations, and therefore a good way to maximize efficiency of NumPy code is to
minimize the number of unique NumPy functions which are called, each of which are
implemented with very high efficiency in :code:`C`. In this case, the inclusion of
:code:`where` as an argument also prevents unnecessary values from being computed in the
first place, only to then be masked out in the immediately subsequent operation.
For these reasons, calling :code:`np.absolute(x, where=mask)` is much more efficient
than calling :code:`np.where(mask, np.absolute(x), np.empty_like(x))` in NumPy.
:func:`ivy.logical_and` is another example where the superset is too much, as we explain
in the extra examples given at the end of this section.

However, other frameworks are able to compile compositions of python operations directly
to computation graphs in low-level languages, and are also able to intelligently fuse
operations into combined kernels, via libraries such as
`TensorRT <https://github.com/NVIDIA/TensorRT>`_.
This removes the need for highly general function signatures such as those found in
NumPy. Instead, a compositional approach is preferred, where each function in Python
generally serves a particular and non-overlapping purpose.
This helps to keep things more clean and clear at the Python level,
without sacrificing efficiency at the lower level.

Balancing Generalization with Efficiency
---------------------------------------

Sometimes, the simplest way to implement superset behaviour comes at the direct expense
of runtime efficiency. We explore this through the examples of :code:`softplus`.

**ivy.softplus**

When looking at the :code:`softplus` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/activations/softplus/softplus_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/math/softplus>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html>`_,
we can see that torch is the only framework which supports the inclusion of the
:code:`beta` and :code:`threshold` arguments, which are added for improved numerical
stability. We can also see that numpy does not support a :code:`softplus`
function at all. Ivy should also support the :code:`beta` and :code:`threshold`
arguments, in order to provide the generalized superset implementation among the backend
frameworks.

Let's take the tensorflow backend implementation as an example when assessing
the necessary changes. Without superset behaviour, the implementation is incredibly
simple, with only a single tensorflow function called under the hood.

.. code-block:: python

    def softplus(x: Tensor,
                 /,
                 *,
                 out: Optional[Tensor] = None) -> Tensor:
        return tf.nn.softplus(x)

The simplest approach would be to implement :code:`softplus` in each Ivy backend as
a simple composition. For example, a simple composition in the tensorflow
backend would look like the following:

.. code-block:: python

    def softplus(x: Tensor,
                 /,
                 *,
                 beta: Optional[Union[int, float]] = 1,
                 threshold: Optional[Union[int, float]] = 20,
                 out: Optional[Tensor] = None) -> Tensor:
        res = (tf.nn.softplus(x * beta)) / beta
        return tf.where(x * beta > threshold, x, res)

This approach uses the default argument values used by PyTorch, and it does indeed
extend the behaviour correctly. However, the implementation now uses **six**
tensorflow function calls instead of one, being: :code:`__mul__`,
:code:`tf.nn.softplus`, :code:`__div__`, :code:`__mul__`, :code:`__gt__`,
:code:`tf.where` in order of execution respectively. If a user doesn't care about the
extra :code:`threshold` and :code:`beta` arguments, then a :code:`6×` increase in
backend functions is a heavy price to pay effiency-wise.

Therefore, we should in general adopt a different approach when implementing superset
behaviour. We should still implement the superset, but keep this extended behaviour as
optional as possible, with maximal effiency and minimal intrusion in the case that this
extended behaviour is not required. The following would be a much better solution:

.. code-block:: python

    def softplus(x: Tensor,
                 /,
                 *,
                 beta: Optional[Union[int, float]] = None,
                 threshold: Optional[Union[int, float]] = None,
                 out: Optional[Tensor] = None) -> Tensor:
        if beta is not None and beta != 1:
            x_beta = x * beta
            res = (tf.nn.softplus(x_beta)) / beta
        else:
            x_beta = x
            res = tf.nn.softplus(x)
        if threshold is not None:
            return tf.where(x_beta > threshold, x, res)
        return res

You will notice that this implementation involves more lines of code, but this should
not be confused with added complexity. All Ivy code should be graph compiled for
efficiency, and in this case all the :code:`if` and :code:`else` statements are removed,
and all that remain are the backend functions which were executed. This new
implementation will compiled to a graph of either one, three, four or six functions
depending on the values of :code:`beta` and :code:`threshold`, while the previous
implementation would *always* compile to six functions.

This does mean we do not adopt the default values used by PyTorch, but that's okay.
Implementing the superset does not mean adopting the same default values for arguments,
it simply means equiping the Ivy function with the capabilities to execute the superset
of behaviours.

More Examples
-------------

We now take a look at some examples, and explain our rational for deciding upon the
function signature that we should use in Ivy. The first three examples are more-or-less
superset examples, while the last example involves a deliberate decision to not
implement the full superset, for some of the reasons explained above.

**ivy.linspace**

When looking at the :code:`linspace` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/creation/linspace/linspace_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linspace.html>`_,
`NumPy <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/linspace>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.linspace.html>`_,
we can see that torch does not support arrays for the :code:`start` and
:code:`end` arguments, while JAX, numpy and tensorflow all do.
Likewise, Ivy also supports arrays for the :code:`start` and :code:`stop` arguments,
and in doing so provides the generalized superset implementation among the backend
frameworks.


**ivy.eye**

When looking at the :code:`eye` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/creation/eye/eye_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.eye.html>`_,
`NumPy <https://numpy.org/devdocs/reference/generated/numpy.eye.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/eye>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.eye.html>`_,
we can see that tensorflow is the only framework which supports a
:code:`batch_shape` argument. Likewise, Ivy also supports a :code:`batch_shape`
argument, and in doing so provides the generalized superset implementation among the
backend frameworks.


**ivy.scatter_nd**

When looking at the :code:`scatter_nd` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/general/scatter_nd/scatter_nd_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at>`_,
`NumPy <https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/scatter_nd>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.scatter.html>`_,
we can see that torch only supports scattering along a single dimension,
while all other frameworks support scattering across multiple dimensions at once.
Likewise, Ivy also supports scattering across multiple dimensions at once,
and in doing so provides the generalized superset implementation among the backend
frameworks.


**ivy.logical_and**

When looking at the :code:`logical_and` (or closest equivalent) implementations for
`Ivy <https://lets-unify.ai/ivy/functional/ivy/elementwise/logical_and/logical_and_functional.html>`_,
`JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.logical_and.html>`_,
`NumPy <https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/math/logical_and>`_,
and
`PyTorch <https://pytorch.org/docs/stable/generated/torch.logical_and.html>`_,
we can see that numpy and torch support the :code:`out` argument for
performing inplace updates, while JAX and tensorflow do not.
With regards to the supported data types, JAX, numpy and torch
support numeric arrays, while tensorflow supports only boolean arrays.
With regards to both of these points, Ivy provides the generalized superset
implementation among the backend frameworks, with support for the :code:`out` argument
and also support for both numeric and boolean arrays in the input.

However, as discussed above, :func:`np.logical_and` also supports the :code:`where`
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


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/_D6xER3H4NU" class="video">
    </iframe>