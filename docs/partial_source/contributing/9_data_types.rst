Data Types
==========

.. _`framework setting`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/framework_handler.py#L205
.. _`_function_w_arrays_dtype_n_dev_handled`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L242
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9
.. _`NON_DTYPE_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L103

All :code:`dtype` arguments are keyword-only.
All creation functions include the :code:`dtype` argument, for specifying the data type of the created array.
Some other non-creation functions also support the :code:`dtype` argument,
such as :code:`ivy.prod` and :code:`ivy.sum`, but most functions do not include it.
The non-creation functions which do support it are generally functions that involve a compounding reduction across the
array, which could result in overflows, and so an explicit :code:`dtype` argument is useful to handling such cases.

The :code:`dtype` argument is handled in `_function_w_arrays_dtype_n_dev_handled`_ for all functions except those
appearing in `NON_WRAPPED_FUNCTIONS`_ or `NON_DTYPE_WRAPPED_FUNCTIONS`_.
As discussed above, this is applied to all applicable function dynamically during `framework setting`_.

Overall, the data type is inferred as follows:

#. if the :code:`dtype` argument is provided, use this directly
#. otherwise, if an array is present in the arguments, set :code:`arr` to this array. \
   This will then be used to infer the data type by calling :code:`ivy.dtype` on the array
#. otherwise, if a *relevant* scalar is present in the arguments, set :code:`arr` to this scalar \
   and derive the data type from this by calling either :code:`ivy.default_int_dtype` or :code:`ivy.default_int_dtype` \
   depending on whether the scalar is an :code:`int` or :code:`float`. \
   This will either return the globally set default :code:`int` or globally set default :code:`float` \
   (settable via :code:`ivy.set_default_int_dtype` and :code:`ivy.set_default_float_dtype` respectively). \
   An example of a *relevant* scalar is :code:`start` in the function :code:`ivy.arange`, \
   which is used to set the starting value of the returned array.\
   Examples of *irrelevant* scalars which should **not** be used for determining the data type are :code:`axis`, \
   :code:`axes`, :code:`dims` etc. which must be integers, and control other configurations of the function \
   being called, with no bearing at all on the data types used by that function.
#. otherwise, if no arrays or relevant scalars are present in the arguments, \
   then use the global default data type, which can either be an :code:`int` or :code:`float` data type. \
   This is settable via :code:`ivy.set_default_dtype`.

For the majority of functions which defer to `_function_w_arrays_dtype_n_dev_handled`_ for handling the data type,
these steps will have been followed and the :code:`dtype` argument will be populated with the correct value
before the backend-specific implementation is even enterred into. Therefore, whereas the :code:`dtype` argument is
listed as optional in the ivy API at :code:`ivy/functional/ivy/category_name.py`,
the argument is listed as required in the backend-specific implementations at
:code:`ivy/functional/backends/backend_name/category_name.py`.

Let's take a look at the function :code:`ivy.prod` as an example.

The implementation in :code:`ivy/functional/ivy/statistical.py` has the following signature:

.. code-block:: python

    def prod(
        x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:

Whereas the backend-specific implementations in :code:`ivy/functional/backends/backend_name/statistical.py`
all list :code:`dtype` as required.

Jax:

.. code-block:: python

    def prod(
        x: JaxArray,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        *,
        dtype: jnp.dtype,
        out: Optional[JaxArray] = None,
    ) -> JaxArray:

MXNet:

.. code-block:: python

    def prod(
        x: mx.nd.NDArray,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        *,
        dtype: type,
        out: Optional[mx.nd.NDArray] = None,
    ) -> mx.nd.NDArray:

NumPy:

.. code-block:: python

    def prod(
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        *,
        dtype: np.dtype,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:

TensorFlow:

.. code-block:: python

    def prod(
        x: Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        *,
        dtype: tf.DType,
        out: Optional[Tensor] = None,
    ) -> Tensor:

PyTorch:

.. code-block:: python

    def prod(
        x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        *,
        dtype: torch.dtype,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

This makes it clear that these backend-specific functions are only enterred into once the correct :code:`dtype`
has been determined.

However, the :code:`dtype` argument for functions listed in `NON_WRAPPED_FUNCTIONS`_ or `NON_DTYPE_WRAPPED_FUNCTIONS`_
are **not** handled by `_function_w_arrays_dtype_n_dev_handled`_,
and so these defaults must be handled by the backend-specific implementations themselves.

One reason for adding a function to `NON_DTYPE_WRAPPED_FUNCTIONS`_ is because it includes *relevant* scalar arguments
for inferring the data type from. `_function_w_arrays_dtype_n_dev_handled`_ is not able to correctly handle such cases,
and so such functions are added to `NON_DTYPE_WRAPPED_FUNCTIONS`_ and the dtype handling is delegated to the
backend-specific implementations.

For example :code:`ivy.full` is listed in `NON_DTYPE_WRAPPED_FUNCTIONS`_ because of the *relevant* :code:`fill_value`
which cannot be correctly handled by `_function_w_arrays_dtype_n_dev_handled`_.

The PyTorch-specific implementation is as follows:

.. code-block:: python

    def full(
        shape: Union[int, Tuple[int]],
        fill_value: Union[int, float],
        *,
        dtype: Optional[Union[ivy.Dtype, torch.dtype]] = None,
        device: torch.device,
    ) -> Tensor:
        return torch.full(
            shape_to_tuple(shape),
            fill_value,
            dtype=ivy.default_dtype(dtype, item=fill_value, as_native=True),
            device=device,
        )

The implementations for all other backends follow a similar pattern to this PyTorch implementation,
where the :code:`dtype` argument is optional and :code:`ivy.default_dtype` is called inside the
backend-specific implementation.
