Inplace Updates
===============

.. _`backend setting`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/framework_handler.py#L205
.. _`_function_w_arrays_n_out_handled`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L166
.. _`torch.tan`: https://pytorch.org/docs/stable/generated/torch.tan.html
.. _`numpy.tan`: https://numpy.org/doc/stable/reference/generated/numpy.tan.html
.. _`tf.math.tan`: https://www.tensorflow.org/api_docs/python/tf/math/tan
.. _`jax.numpy.tan`: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tan.html?highlight=tan
.. _`mx.nd.tan`: https://mxnet.apache.org/versions/1.6/api/r/docs/api/mx.nd.tan.html
.. _`presence of this argument`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L154
.. _`by the backend function`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L199
.. _`by the wrapper`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L203
.. _`handled by the wrapper`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/func_wrapper.py#L210
.. _`_wrap_fn`: https://github.com/unifyai/ivy/blob/6497b8a3d6b0d8aac735a158cd03c8f98eb288c2/ivy/container/wrapping.py#L69
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9

All Ivy functions which return a single array should support inplace updates, with the inclusion of a keyword-only
:code:`out` argument, with type hint :code:`Optional[Union[ivy.Array, ivy.Container]]` for *flexible* functions
and :code:`Optional[ivy.Array]` otherwise.

When this argument is unspecified, then the return is simply provided in a newly created :code:`ivy.Array` or
:code:`ivy.Container`. However, when :code:`out` is specified, then the return is provided as an inplace update of the
:code:`out` argument provided. This can for example be the same as the input to the function,
resulting in a simple inplace update of the input.

In the case of :code:`ivy.Array` return types, the :code:`out` argument is predominatly handled in
`_function_w_arrays_n_out_handled`_, which as discussed above,
is also responsible for converting :code:`ivy.Array` instances to :code:`ivy.NativeArray`
instances before calling the backend function, and then back to :code:`ivy.Array` instances again for returning.
As explained above, this wrapping is applied to every function (except those appearing in `NON_WRAPPED_FUNCTIONS`_)
dynamically during `backend setting`_.

However, `_function_w_arrays_n_out_handled`_ does not handle backend-specific functions which support an :code:`out`
argument directly, such as `torch.tan`_ and `numpy.tan`_.
When implementing backend-specific functions, the :code:`out` argument should only be added to functions which wrap a
function in the backend supporting inplace updates directly.
`tf.math.tan`_, `jax.numpy.tan`_ and `mx.nd.tan`_ for example do **not** support inplace updates,
and so the :code:`out` argument should **not** be included in these backend-specific :code:`tan` implementations.

The implementations of :code:`ivy.tan` for each backend are as follows.

Jax (no :code:`out` argument):

.. code-block:: python

    def tan(x: JaxArray) -> JaxArray:
        return jnp.tan(x)

MXNet (no :code:`out` argument):

.. code-block:: python

    def tan(x: mx.NDArray) -> mx.NDArray:
        return mx.nd.tan(x)

NumPy (includes :code:`out` argument):

.. code-block:: python

    def tan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.tan(x, out=out)

TensorFlow (no :code:`out` argument):

.. code-block:: python

    def tan(x: Tensor) -> Tensor:
        return tf.tan(x)

PyTorch (includes :code:`out` argument):

.. code-block:: python

    def tan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.tan(x, out=out)


It's very important to remove the :code:`out` argument from backend implementations that do not actually handle it,
as the `presence of this argument`_ dictates whether the argument should be handled
`by the backend function`_ or `by the wrapper`_.

This distinction only concerns how the inplace update is applied to the native array,
which is operated upon directly by the backend.
If :code:`out` is specified, an inplace update is always **also** performed on the :code:`ivy.Array` instance itself,
which is how :code:`out` is provided to the function. This inplace update is always `handled by the wrapper`_.

Alternatively, if :code:`out` is an :code:`ivy.Container`, then the inplace update is always handled by `_wrap_fn`_ in
the container wrapping module.
