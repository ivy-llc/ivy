Handling Devices
================

.. _`framework setting`: https://github.com/unifyai/ivy/blob/ee0da7d142ba690a317a4fe00a4dd43cf8634642/ivy/framework_handler.py#L205
.. _`_function_w_arrays_dtype_n_dev_handled`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L242
.. _`NON_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L9
.. _`NON_DTYPE_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L103
.. _`NON_DEV_WRAPPED_FUNCTIONS`: https://github.com/unifyai/ivy/blob/fdaea62380c9892e679eba37f26c14a7333013fe/ivy/func_wrapper.py#L104

Like with :code:`dtype`, all :code:`device` arguments are also keyword-only.
All creation functions include the :code:`device` argument,
for specifying the device on which to place the created array.
Some other functions outside of the :code:`creation.py` submodule also support the :code:`device` argument,
such as :code:`ivy.random_uniform` which is located in :code:`random.py`,
but this is simply because of dual categorization.
:code:`ivy.random_uniform` is also essentially a creation function,
despite not being being located in :code:`creation.py`.

The :code:`device` argument is generally not included for functions which accept arrays in the input and perform
operations on these arrays. In such cases, the device of the output arrays is the same as the device for
the input arrays. In cases where the input arrays are located on different devices, an error will be thrown.

The :code:`device` argument is handled in `_function_w_arrays_dtype_n_dev_handled`_ for all functions except those
appearing in `NON_WRAPPED_FUNCTIONS`_ or `NON_DEV_WRAPPED_FUNCTIONS`_.
This is similar to how :code:`dtype` is handled,
with the exception that functions are omitted if they're in `NON_DEV_WRAPPED_FUNCTIONS`_ in this case rather than
`NON_DTYPE_WRAPPED_FUNCTIONS`_. As discussed above,
this is applied to all applicable function dynamically during `framework setting`_.

Overall, the device is inferred as follows:

#. if the :code:`device` argument is provided, use this directly
#. otherwise, if an array is present in the arguments (very rare if :code:`device` is present), \
   set :code:`arr` to this array. This will then be used to infer the device by calling :code:`ivy.dev` on the array
#. otherwise, if no arrays are present in the arguments (by far the most common case if :code:`device` is present), \
   then use the global default device, \
   which currently can either be :code:`cpu` or :code:`gpu:idx`, \
   but more device types and multi-node configurations are in the pipeline. \
   The default device is settable via :code:`ivy.set_default_device`.

For the majority of functions which defer to `_function_w_arrays_dtype_n_dev_handled`_ for handling the device,
these steps will have been followed and the :code:`device` argument will be populated with the correct value
before the framework-specific implementation is even enterred into. Therefore, whereas the :code:`device` argument is
listed as optional in the ivy API at :code:`ivy/functional/ivy/category_name.py`,
the argument is listed as required in the framework-specific implementations at
:code:`ivy/functional/backends/backend_name/category_name.py`.

This is exactly the same as with the :code:`dtype` argument, which is explained above.

Let's take a look at the function :code:`ivy.zeros` as an example.

The implementation in :code:`ivy/functional/ivy/creation.py` has the following signature:

.. code-block:: python

    def zeros(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Array:

Whereas the framework-specific implementations in :code:`ivy/functional/backends/backend_name/creation.py`
all list :code:`device` as required.

Jax:

.. code-block:: python

    def zeros(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: jnp.dtype,
        device: jaxlib.xla_extension.Device,
    ) -> JaxArray:

MXNet:

.. code-block:: python

    def zeros(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: type,
        device: mx.context.Context,
    ) -> mx.nd.NDArray:

NumPy:

.. code-block:: python

    def zeros(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: np.dtype,
        device: str,
    ) -> np.ndarray:

TensorFlow:

.. code-block:: python

    def zeros(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: tf.DType,
        device: str,
    ) -> Tensor:

PyTorch:

.. code-block:: python

    def zeros(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:

This makes it clear that these framework-specific functions are only enterred into once the correct :code:`device`
has been determined.

However, the :code:`device` argument for functions listed in `NON_WRAPPED_FUNCTIONS`_ or `NON_DEV_WRAPPED_FUNCTIONS`_
are **not** handled by `_function_w_arrays_dtype_n_dev_handled`_,
and so these defaults must be handled by the framework-specific implementations themselves,
by calling :code:`ivy.default_device` internally.