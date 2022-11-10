Inplace Updates
===============

.. _`backend setting`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`handle_out_argument`: https://github.com/unifyai/ivy/blob/dcfec8b85de3c422dc0ca1970d67cb620cae62a4/ivy/func_wrapper.py#L340
.. _`torch.tan`: https://pytorch.org/docs/stable/generated/torch.tan.html
.. _`numpy.tan`: https://numpy.org/doc/stable/reference/generated/numpy.tan.html
.. _`tf.math.tan`: https://www.tensorflow.org/api_docs/python/tf/math/tan
.. _`jax.numpy.tan`: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tan.html?highlight=tan
.. _`mx.nd.tan`: https://mxnet.apache.org/versions/1.6/api/r/docs/api/mx.nd.tan.html
.. _`presence of this attribute`: https://github.com/unifyai/ivy/blob/8ded4a5fc13a278bcbf2d76d1fa58ab41f5797d0/ivy/func_wrapper.py#L341
.. _`by the backend function`: https://github.com/unifyai/ivy/blob/8ded4a5fc13a278bcbf2d76d1fa58ab41f5797d0/ivy/func_wrapper.py#L372
.. _`by the wrapper`: https://github.com/unifyai/ivy/blob/8ded4a5fc13a278bcbf2d76d1fa58ab41f5797d0/ivy/func_wrapper.py#L377
.. _`handled by the wrapper`: https://github.com/unifyai/ivy/blob/8ded4a5fc13a278bcbf2d76d1fa58ab41f5797d0/ivy/func_wrapper.py#L373
.. _`_wrap_fn`: https://github.com/unifyai/ivy/blob/6497b8a3d6b0d8aac735a158cd03c8f98eb288c2/ivy/container/wrapping.py#L69
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9
.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`ivy.reshape`: https://github.com/unifyai/ivy/blob/633eb420c5006a0a17c238bfa794cf5b6add8598/ivy/functional/ivy/manipulation.py#L418
.. _`ivy.astype`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/data_type.py#L46
.. _`ivy.asarray`: https://github.com/unifyai/ivy/blob/8482eb3fcadd0721f339a1a55c3f3b9f5c86d8ba/ivy/functional/ivy/creation.py#L114
.. _`wrapping`:
.. _`ivy.inplace_update`: https://github.com/unifyai/ivy/blob/3a21a6bef52b93989f2fa2fa90e3b0f08cc2eb1b/ivy/functional/ivy/general.py#L1137
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`inplace updates channel`: https://discord.com/channels/799879767196958751/982738152236130335
.. _`inplace updates forum`: https://discord.com/channels/799879767196958751/1028681672268464199
.. _`in the decorator`: https://github.com/unifyai/ivy/blob/588618fe04de21f79d68a8f6cbb48ab3402c6905/ivy/func_wrapper.py#L287

Inplace updates enable users to overwrite the contents of existing arrays with new data.
This enables much more control over the memory-efficiency of the program, preventing old unused arrays from being kept in memory for any longer than is strictly necessary.

The function `ivy.inplace_update`_ enables explicit inplace updates.
:func:`ivy.inplace_update` is a *primary* function, and the backend-specific implementations for each backend are presented below.
We also explain the rational for why each implementation is the way it is, and the important differences.

This is one particular area of the Ivy code where, technically speaking, the function :func:`ivy.inplace_update` will result in subtly different behaviour for each backend, unless the :code:`ensure_in_backend` flag is set to :code:`True`.

While :class:`ivy.Array` instances will always be inplace updated consistently, in some cases it is simply not possible to also inplace update the :class:`ivy.NativeArray` which :class:`ivy.Array` wraps, due to design choices made by each backend.

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

JAX **does not** natively support inplace updates, and so there is no way of actually inplace updating the :code:`JaxArray` instance :code:`x_native`.
Therefore, an inplace update is only performed on :class:`ivy.Array` instances provided in the input.

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

NumPy **does** natively support inplace updates, and so :code:`x_native` is updated inplace with :code:`val_native`.
Following this, an inplace update is then also performed on the :class:`ivy.Array` instance, if provided in the input.

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

TensorFlow **does not** natively support inplace updates for :class:`tf.Tensor` instances, and in such cases so there is no way of actually inplace updating the :class:`tf.Tensor` instance :code:`x_native`.
However, TensorFlow **does** natively support inplace updates for :class:`tf.Variable` instances.
Therefore, if :code:`x_native` is a :class:`tf.Variable`, then :code:`x_native` is updated inplace with :code:`val_native`.
Irrespective of whether the native array is a :class:`tf.Tensor` or a :class:`tf.Variable`, an inplace update is then also performed on the :class:`ivy.Array` instance, if provided in the input.

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

PyTorch **does** natively support inplace updates, and so :code:`x_native` is updated inplace with :code:`val_native`.
Following this, an inplace update is then also performed on the :class:`ivy.Array` instance, if provided in the input.

The function :func:`ivy.inplace_update` is also *nestable*, meaning it can accept :class:`ivy.Container` instances in the input.
If an :class:`ivy.Container` instance is provided for the argument :code:`x`, then along with the arrays at all of the leaves, the container :code:`x` is **also** inplace updated, meaning that a new :class:`ivy.Container` instance is not created for the function return.

out argument
------------

Most functions in Ivy support inplace updates via the inclusion of a keyword-only :code:`out` argument.
This enables users to specify the array to which they would like the output of a function to be written.
This could for example be the input array itself, but can also be any other array of choice.

All Ivy functions which return a single array should support inplace updates via the :code:`out` argument.
The type hint of the :code:`out` argument is :code:`Optional[ivy.Array]`.
However, as discussed above, if the function is *nestable* then :class:`ivy.Container` instances are also supported.
:class:`ivy.Container` is omitted from the type hint in such cases, as explained in the :ref:`Function Arguments` section.

When the :code:`out` argument is unspecified, then the return is simply provided in a newly created :class:`ivy.Array` (or :class:`ivy.Container` if *nestable*).
However, when :code:`out` is specified, then the return is provided as an inplace update of the :code:`out` argument provided.
This can for example be the same as the input to the function, resulting in a simple inplace update of the input.

In the case of :class:`ivy.Array` return types, the :code:`out` argument is predominantly handled in `handle_out_argument`_.
As explained in the :ref:`Function Wrapping` section, this wrapping is applied to every function with the :code:`@handle_out_argument` decorator dynamically during `backend setting`_.

**Primary Functions**

In the case of *primary* functions, `handle_out_argument`_ does not handle the backend-specific inplace updates in cases where the backend function being wrapped supports them directly, such as `torch.tan`_ and `numpy.tan`_, which both support the :code:`out` argument directly.
When implementing backend-specific functions, the attribute :code:`support_native_out` should be added to all functions which wrap a function in the backend supporting inplace updates directly.
`tf.math.tan`_, `jax.numpy.tan`_ and `mx.nd.tan`_ for example do **not** support inplace updates, and so the :code:`support_native_out` attribute should **not** be added to the :code:`tan` implementations.

The implementations of :func:`ivy.tan` for each backend are as follows.

**JAX** (no :code:`support_native_out` attribute):

.. code-block:: python

    def tan(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
        return jnp.tan(x)

**NumPy** (includes :code:`support_native_out` attribute):

.. code-block:: python

    @_handle_0_dim_output
    def tan(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
        return np.tan(x, out=out)

    tan.support_native_out = True

**TensorFlow** (no :code:`support_native_out` attribute):

.. code-block:: python

    def tan(
        x: Union[tf.Tensor, tf.Variable],
        /,
        *,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    ) -> Union[tf.Tensor, tf.Variable]:
        return tf.tan(x)

**PyTorch** (includes :code:`support_native_out` attribute):

.. code-block:: python

    def tan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.tan(x, out=out)

    tan.support_native_out = True

It's very important to ensure the :code:`support_native_out` attribute is not added to backend implementations that do not handle the :code:`out` argument, as the `presence of this attribute`_ dictates whether the argument should be handled `by the backend function`_ or `by the wrapper`_.

This distinction only concerns how the inplace update is applied to the native array, which is operated upon directly by the backend.
If :code:`out` is specified in an Ivy function, then an inplace update is always **also** performed on the :class:`ivy.Array` instance itself, which is how :code:`out` is provided to the function originally.
The inplace update of this :class:`ivy.Array` is always `handled by the wrapper`_.

Alternatively, if :code:`out` is an :class:`ivy.Container`, then the inplace update is always handled by `_wrap_fn`_ in the container wrapping module.

**Special Case**

Take a function which has multiple possible "paths" through the code:

.. code-block:: python

    def cholesky(
        x: torch.Tensor, /, *, upper: bool = False, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not upper:
            return torch.linalg.cholesky(x, out=out)
        else:
            ret = torch.transpose(
                torch.linalg.cholesky(
                    torch.transpose(x, dim0=len(x.shape) - 1, dim1=len(x.shape) - 2)
                ),
                dim0=len(x.shape) - 1,
                dim1=len(x.shape) - 2,
            )
            if ivy.exists(out):
                return ivy.inplace_update(out, ret)
            return ret


    cholesky.support_native_out = True

Here we still have the :attr:`support_native_out` attribute since we want to take advantage of the native inplace update enabled by :func:`torch.linalg.cholesky` in the first condition.
However, in the :code:`else` statement, the last operation is :func:`torch.transpose` which does not support the :code:`out` argument, and so the native inplace update can't be performed by torch here.
This is why we need to call :func:`ivy.inplace_update` explicitly here, to ensure the native inplace update is performed, as well as the :class:`ivy.Array` inplace update.

**Compositional Functions**

For *compositional* functions, the :code:`out` argument should **always** be handled in the compositional implementation, with no wrapping applied at all.
This is for a few reasons:

#. we need to show the :code:`out` argument in the compositional function signature, as this is the only function implementation in the codebase.
   Adding an argument unused in the implementation could cause some confusion.
#. generally, inplace updates are performed because memory management is an area of concern for the user.
   By handling the :code:`out` argument in the compositional implementation itself.
   We can maximize the memory efficiency of the function, using inplace updates in as many of the inner Ivy functions as possible.
#. this enables us to make use of backend-specific :code:`out` argument handling.

The second and third points are the most important points.

We'll use :func:`ivy.cross_entropy` as an example:

.. code-block:: python

    def cross_entropy(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = -1,
        epsilon: float =1e-7,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        log_pred = ivy.log(pred)
        return ivy.negative(ivy.sum(log_pred * true, axis, out=out), out=out)

By handling the :code:`out` argument in the function, we are able to get the benefits outlined above.
Firstly, the return of :func:`ivy.sum` is the same shape and type as the return of the entire function, and so we can also write this output to the :code:`out` argument inplace.
We can then subsequently overwrite the contents of :code:`out` again with the return of the :func:`ivy.negative` function.
This minimizes the number of arrays created during the execution of the function, which is generally the intention when specifying the :code:`out` argument.
Additionally, with a PyTorch backend, the :func:`ivy.negative` function defers to the :code:`out` argument of :func:`torch.negative` function directly, which is the most efficient inplace update possible, making use of backend-specific optimizations.

If we had instead simply used the wrapper `handle_out_argument`_, then we would not leverage any of these benefits, and instead simply call :func:`ivy.inplace_update` at the very end of the function call.

For some compositional functions, the internal function which generates the final return value does not itself support the :code:`out` argument.
For example, `ivy.multi_head_attention <https://github.com/unifyai/ivy/blob/2045db570d7977830681a7498a3c1045fb5bcc79/ivy/functional/ivy/layers.py#L165>`_ includes support for arbitrary functions passed in the input, including :code:`to_out_fn` which, if specified, is applied to the outputs before returning.
For such functions, the inplace update should just be performed using :func:`ivy.inplace_update` at the end of the function, like `so <https://github.com/unifyai/ivy/blob/2045db570d7977830681a7498a3c1045fb5bcc79/ivy/functional/ivy/layers.py#L254>`_.

Technically, this could be handled using the `handle_out_argument`_ wrapping, but we opt to implement this in the compositional function itself, due to point 1 mentioned above.

**Mixed Functions**

As explained in the :ref:`Function Types` section, *mixed* functions can effectively behave as either compositional or primary functions, depending on the backend that is selected.

Unlike *compositional* arguments, where the :code:`handle_out_argument` decorator is not included, this decorator *should* be included for *mixed* functions.
This decorator is needed in order to ensure the :code:`out` argument is handled correctly when the backend *does* include a backend-specific implementation, which itself may or may not handle the :code:`out` argument explicitly.
In such cases, the *mixed* function behaves like a *primary* function.
If the backend-specific implementation does not handle the :code:`out` argument explicitly (there is no attribute :code:`support_native_out` specified on the backend function), then it will need to be handled `in the decorator`_.

However, the inclusion of this decorator means that in cases where the *mixed* function is called compositionally (there is no backend implementation), then the :code:`out` argument will also be handled `in the decorator`_, this time because of the lack of the :code:`support_native_out` attribute found on the compositional implementation.
But this is not ideal.
All compositional implementations are fully capable of handling the :code:`out` argument explicitly, and so handling it `in the decorator`_ will likely be less efficient, and prevent us from leveraging backend-specific in-place optimizations where they might exist when calling the individual Ivy functions of the compositional implementation.

Therefore, we always add the :code:`support_native_out` attribute to *mixed* functions, to ensure that the :code:`out` argument is always handled directly by the compositional implementation, rather than being handled `in the decorator`_.

copy argument
-------------

As well as the :code:`out` argument, a few functions also support the :code:`copy` argument.
The functions with support for the :code:`copy` argument are all in the `Array API Standard`_, and the standard mandates the inclusion of :code:`copy` in each case.
These functions are: `ivy.reshape`_ (`in the standard <https://github.com/data-apis/array-api/blob/5ba86db7ff5f9ddd9e956808c3659b1fc7f714cc/spec/API_specification/array_api/manipulation_functions.py#L106>`_), `ivy.astype`_ (`in the standard <https://github.com/data-apis/array-api/blob/5ba86db7ff5f9ddd9e956808c3659b1fc7f714cc/spec/API_specification/array_api/data_type_functions.py#L3>`_) and `ivy.asarray`_ (`in the standard <https://github.com/data-apis/array-api/blob/5ba86db7ff5f9ddd9e956808c3659b1fc7f714cc/spec/API_specification/array_api/creation_functions.py#L31>`_).

The :code:`copy` argument dictates whether a new copy should be created, or whether the input array should be updated inplace.
When :code:`copy` is not specified explicitly, then an inplace update is performed with the same behaviour as :code:`copy=False`.
Setting :code:`copy=False` is equivalent to passing :code:`out=input_array`.
If only one of :code:`copy` or :code:`out` is specified, then this specified argument is given priority.
If both are specified, then priority is given to the more general :code:`out` argument.
As with the :code:`out` argument, the :code:`copy` argument is also handled `by the wrapper <insert_link>`_.

**Round Up**

This should have hopefully given you a good feel for inplace updates, and how these are handled in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `inplace updates channel`_ or in the `inplace updates forum`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/n8ko-Ig2eZ0" class="video">
    </iframe>
