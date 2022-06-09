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
.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`ivy.reshape`: https://github.com/unifyai/ivy/blob/633eb420c5006a0a17c238bfa794cf5b6add8598/ivy/functional/ivy/manipulation.py#L418
.. _`ivy.astype`: https://github.com/unifyai/ivy/blob/633eb420c5006a0a17c238bfa794cf5b6add8598/ivy/functional/ivy/data_type.py#L164
.. _`ivy.asarray`: https://github.com/unifyai/ivy/blob/633eb420c5006a0a17c238bfa794cf5b6add8598/ivy/functional/ivy/creation.py#L64
.. _`wrapping`:
.. _`ivy.inplace_update`: https://github.com/unifyai/ivy/blob/3a21a6bef52b93989f2fa2fa90e3b0f08cc2eb1b/ivy/functional/ivy/general.py#L1137
.. _`inplace updates discussion`: https://github.com/unifyai/ivy/discussions/1319
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`inplace updates channel`: https://discord.com/channels/799879767196958751/982738152236130335

Inplace updates enable users to overwrite the contents of existing arrays with new data.
This enables much more control over the memory-efficiency of the program,
preventing old unused arrays from being kept in memory for any longer than is strictly necessary.

The function `ivy.inplace_update`_ enables explicit inplace updates.
:code:`ivy.inplace_update` is a *primary* function,
and the backend-specific implementations for each backend are presented below.
We also explain the rational for why each implementation is the way it is,
and the important differences.

This is one particular area of the Ivy code where, technically speaking,
the function :code:`ivy.inplace_update` will result in subtly different behaviour
for each backend, unless the :code:`ensure_in_backend` flag is set to :code:`True`.

While :code:`ivy.Array` instances will always be inplace updated consistently,
in some cases it is simply not possible to also inplace update the :code:`ivy.NativeArray`
which :code:`ivy.Array` wraps, due to design choices made by each backend.

**JAX**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, JaxArray],
        val: Union[ivy.Array, JaxArray],
        ensure_in_backend: bool = False,
    ) -> ivy.Array:
        if ensure_in_backend:
            raise Exception("JAX does not natively support inplace updates")
        (x_native, val_native), _ = ivy.args_to_native(x, val)
        if ivy.is_ivy_array(x):
            x.data = val_native
        else:
            raise Exception("JAX does not natively support inplace updates")
        return x

JAX **does not** natively support inplace updates,
and so there is no way of actually inplace updating the :code:`JaxArray` instance :code:`x_native`.
Therefore, an inplace update is only performed on :code:`ivy.Array` instances provided in the input.

**MXNet**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, mx.nd.NDArray],
        val: Union[ivy.Array, mx.nd.NDArray],
        ensure_in_backend: bool = False,
    ) -> ivy.Array:
        (x_native, val_native), _ = ivy.args_to_native(x, val)
        x_native[:] = val_native
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
        return x

MXNet **does** natively support inplace updates,
and so :code:`x_native` is updated inplace with :code:`val_native`.
Following this, an inplace update is then also performed on the :code:`ivy.Array` instance, if provided in the input.

**NumPy**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, np.ndarray],
        val: Union[ivy.Array, np.ndarray],
        ensure_in_backend: bool = False,
    ) -> ivy.Array:
        (x_native, val_native), _ = ivy.args_to_native(x, val)
        x_native.data = val_native
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
        return x

NumPy **does** natively support inplace updates,
and so :code:`x_native` is updated inplace with :code:`val_native`.
Following this, an inplace update is then also performed on the :code:`ivy.Array` instance,
if provided in the input.

**TensorFlow**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, tf.Tensor, tf.Variable],
        val: Union[ivy.Array, tf.Tensor, tf.Variable],
        ensure_in_backend: bool = False,
    ) -> ivy.Array:
        (x_native, val_native), _ = ivy.args_to_native(x, val)
        if ivy.is_variable(x_native):
            x_native.assign(val_native)
            if ivy.is_ivy_array(x):
                x.data = x_native
            else:
                x = ivy.Array(x_native)
        elif ensure_in_backend:
            raise Exception("TensorFlow does not support inplace updates of the tf.Tensor")
        elif ivy.is_ivy_array(x):
            x.data = val_native
        else:
            raise Exception("TensorFlow does not support inplace updates of the tf.Tensor")
        return x

TensorFlow **does not** natively support inplace updates for :code:`tf.Tensor` instances,
and in such cases so there is no way of actually inplace updating the :code:`tf.Tensor` instance :code:`x_native`.
However, TensorFlow **does** natively support inplace updates for :code:`tf.Variable` instances.
Therefore, if :code:`x_native` is a :code:`tf.Variable`,
then :code:`x_native` is updated inplace with :code:`val_native`.
Irrespective of whether the native array is a :code:`tf.Tensor` or a :code:`tf.Variable`,
an inplace update is then also performed on the :code:`ivy.Array` instance, if provided in the input.

**PyTorch**:

.. code-block:: python

    def inplace_update(
        x: Union[ivy.Array, torch.Tensor],
        val: Union[ivy.Array, torch.Tensor],
        ensure_in_backend: bool = False,
    ) -> ivy.Array:
        (x_native, val_native), _ = ivy.args_to_native(x, val)
        x_native.data = val_native
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
        return x

PyTorch **does** natively support inplace updates,
and so :code:`x_native` is updated inplace with :code:`val_native`.
Following this, an inplace update is then also performed on the :code:`ivy.Array` instance,
if provided in the input.

The function :code:`ivy.inplace_update` is also *nestable*,
meaning it can accept :code:`ivy.Container` instances in the input.
If an :code:`ivy.Container` instance is provided for the argument :code:`x`,
then along with the arrays at all of the leaves,
the container :code:`x` is **also** inplace updated,
meaning that a new :code:`ivy.Container` instance is not created for the function return.

out argument
------------

Most functions in Ivy support inplace updates via the inclusion of a keyword-only :code:`out` argument.
This enables users to specify the array to which they would like the output of a function to be written.
This could for example be the input array itself, but can also be any other array of choice.

All Ivy functions which return a single array should support inplace updates.
The type hint of the :code:`out` argument is :code:`Optional[ivy.Array]`.
However, as discussed above, the function is *nestable*, meaning :code:`ivy.Container` instances are also supported.
This is omitted from the type hint, as explained in the :ref:`Function Arguments` section.

When the :code:`out` argument is unspecified, then the return is simply provided in a newly created :code:`ivy.Array` or
:code:`ivy.Container`. However, when :code:`out` is specified, then the return is provided as an inplace update of the
:code:`out` argument provided. This can for example be the same as the input to the function,
resulting in a simple inplace update of the input.

In the case of :code:`ivy.Array` return types, the :code:`out` argument is predominantly handled in
`_function_w_arrays_n_out_handled`_, which as discussed in the :ref:`Arrays` section,
is also responsible for converting :code:`ivy.Array` instances to :code:`ivy.NativeArray`
instances before calling the backend function, and then back to :code:`ivy.Array` instances again for returning.
As explained in the :ref:`Function Wrapping` section,
this wrapping is applied to every function (except those appearing in `NON_WRAPPED_FUNCTIONS`_)
dynamically during `backend setting`_.

However, `_function_w_arrays_n_out_handled`_ does not handle backend-specific functions which support an :code:`out`
argument directly, such as `torch.tan`_ and `numpy.tan`_.
When implementing backend-specific functions, the :code:`out` argument should only be added to functions which wrap a
function in the backend supporting inplace updates directly.
`tf.math.tan`_, `jax.numpy.tan`_ and `mx.nd.tan`_ for example do **not** support inplace updates,
and so the :code:`out` argument should **not** be included in these backend-specific :code:`tan` implementations.

The implementations of :code:`ivy.tan` for each backend are as follows.

**JAX** (no :code:`out` argument):

.. code-block:: python

    def tan(x: JaxArray) -> JaxArray:
        return jnp.tan(x)

**MXNet** (no :code:`out` argument):

.. code-block:: python

    def tan(x: mx.NDArray) -> mx.NDArray:
        return mx.nd.tan(x)

**NumPy** (includes :code:`out` argument):

.. code-block:: python

    def tan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.tan(x, out=out)

**TensorFlow** (no :code:`out` argument):

.. code-block:: python

    def tan(x: Tensor) -> Tensor:
        return tf.tan(x)

**PyTorch** (includes :code:`out` argument):

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

copy argument
-------------

As well as the :code:`out` argument, a few functions also support the :code:`copy` argument.
The functions with support for the :code:`copy` argument are all in the `Array API Standard`_,
and the standard mandates the inclusion of :code:`copy` in each case.
These functions are:
`ivy.reshape`_ (`in the standard <https://github.com/data-apis/array-api/blob/5ba86db7ff5f9ddd9e956808c3659b1fc7f714cc/spec/API_specification/array_api/manipulation_functions.py#L106>`_),
`ivy.astype`_ (`in the standard <https://github.com/data-apis/array-api/blob/5ba86db7ff5f9ddd9e956808c3659b1fc7f714cc/spec/API_specification/array_api/data_type_functions.py#L3>`_)
and `ivy.asarray`_ (`in the standard <https://github.com/data-apis/array-api/blob/5ba86db7ff5f9ddd9e956808c3659b1fc7f714cc/spec/API_specification/array_api/creation_functions.py#L31>`_).

The :code:`copy` argument dictates whether a new copy should be created,
or whether the input array should be updated inplace.
When :code:`copy` is not specified explicitly, then an inplace update is performed
with the same behaviour as :code:`copy=False`.
Setting :code:`copy=False` is equivalent to passing :code:`out=input_array`.
If only one of :code:`copy` or :code:`out` is specified, then this specified argument is given priority.
If both are specified, then priority is given to the more general :code:`out` argument.
As with the :code:`out` argument, the :code:`copy` argument is also handled `by the wrapper <insert_link>`_

**Round Up**

This should have hopefully given you a good feel for inplace updates, and how these are handled in Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `inplace updates discussion`_,
or reach out on `discord`_ in the `inplace updates channel`_!
