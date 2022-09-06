Devices
=======

.. _`backend setting`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/backend_handler.py#L204
.. _`infer_device`: https://github.com/unifyai/ivy/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/ivy/func_wrapper.py#L286
.. _`ivy.Device`: https://github.com/unifyai/ivy/blob/0b89c7fa050db13ef52b0d2a3e1a5fb801a19fa2/ivy/__init__.py#L42
.. _`empty class`: https://github.com/unifyai/ivy/blob/0b89c7fa050db13ef52b0d2a3e1a5fb801a19fa2/ivy/__init__.py#L34
.. _`device class`: https://github.com/unifyai/ivy/blob/0b89c7fa050db13ef52b0d2a3e1a5fb801a19fa2/ivy/functional/backends/torch/__init__.py#L13
.. _`device.py`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/device.py
.. _`ivy.total_mem_on_dev`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/device.py#L460
.. _`ivy.dev_util`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/device.py#L600
.. _`ivy.num_cpu_cores`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/device.py#L659
.. _`ivy.default_device`: https://github.com/unifyai/ivy/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/ivy/functional/ivy/device.py#L720
.. _`devices discussion`: https://github.com/unifyai/ivy/discussions/1317
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`devices channel`: https://discord.com/channels/799879767196958751/982738108166602752

The devices currently supported by Ivy are as follows:

* cpu
* gpu:idx
* tpu:idx

In a similar manner to the :code:`ivy.Dtype` and :code:`ivy.NativeDtype` classes (see :ref:`Data Types`),
there is both an `ivy.Device`_ class and an :code:`ivy.NativeDevice` class,
with :code:`ivy.NativeDevice` initially set as an `empty class`_.
The :code:`ivy.Device` class derives from :code:`str`,
and has simple logic in the constructor to verify that the string formatting is correct.
When a backend is set, the :code:`ivy.NativeDtype` is replaced with the backend-specific `device class`_.

Device Module
-------------

The `device.py`_ module provides a variety of functions for working with devices.
A few examples include
:code:`ivy.get_all_ivy_arrays_on_dev` which gets all arrays which are currently alive on the specified device,
:code:`ivy.dev` which gets the device for input array,
and :code:`ivy.num_gpus` which determines the number of available GPUs for use with the backend framework.

Many functions in the :code:`device.py` module are *convenience* functions,
which means that they do not directly modify arrays,
as explained in the :ref:`Function Types` section.

For example, the following are all convenience functions:
`ivy.total_mem_on_dev`_, which gets the total amount of memory for a given device,
`ivy.dev_util`_, which gets the current utilization (%) for a given device,
`ivy.num_cpu_cores`_, which determines the number of cores available in the CPU,
and `ivy.default_device`_, which returns the correct device to use.

`ivy.default_device`_ is arguably the most important function.
Any function in the functional API that receives a :code:`device` argument will make use of this function,
as explained below.

Arguments in other Functions
----------------------------

Like with :code:`dtype`, all :code:`device` arguments are also keyword-only.
All creation functions include the :code:`device` argument,
for specifying the device on which to place the created array.
Some other functions outside of the :code:`creation.py` submodule also support the :code:`device` argument,
such as :code:`ivy.random_uniform` which is located in :code:`random.py`,
but this is simply because of dual categorization.
:code:`ivy.random_uniform` is also essentially a creation function,
despite not being located in :code:`creation.py`.

The :code:`device` argument is generally not included for functions which accept arrays in the input and perform
operations on these arrays. In such cases, the device of the output arrays is the same as the device for
the input arrays. In cases where the input arrays are located on different devices, an error will generally be thrown,
unless the function is specific to distributed training.

The :code:`device` argument is handled in `infer_device`_ for all functions which have the :code:`@infer_device`
decorator, similar to how :code:`dtype` is handled.
This function calls `ivy.default_device`_ in order to determine the correct device.
As discussed in the :ref:`Function Wrapping` section,
this is applied to all applicable functions dynamically during `backend setting`_.

Overall, `ivy.default_device`_ infers the device as follows:

#. if the :code:`device` argument is provided, use this directly
#. otherwise, if an array is present in the arguments (very rare if the :code:`device` argument is present), \
   set :code:`arr` to this array. This will then be used to infer the device by calling :code:`ivy.dev` on the array
#. otherwise, if no arrays are present in the arguments (by far the most common case if the :code:`device` argument is present), \
   then use the global default device, \
   which currently can either be :code:`cpu`, :code:`gpu:idx` or :code:`tpu:idx`. \
   The default device is settable via :code:`ivy.set_default_device`.

For the majority of functions which defer to `infer_device`_ for handling the device,
these steps will have been followed and the :code:`device` argument will be populated with the correct value
before the backend-specific implementation is even entered into. Therefore, whereas the :code:`device` argument is
listed as optional in the ivy API at :code:`ivy/functional/ivy/category_name.py`,
the argument is listed as required in the backend-specific implementations at
:code:`ivy/functional/backends/backend_name/category_name.py`.

This is exactly the same as with the :code:`dtype` argument, as explained in the :ref:`Data Types` section.

Let's take a look at the function :code:`ivy.zeros` as an example.

The implementation in :code:`ivy/functional/ivy/creation.py` has the following signature:

.. code-block:: python

    @outputs_to_ivy_arrays
    @handle_out_argument
    @infer_dtype
    @infer_device
    def zeros(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Array:

Whereas the backend-specific implementations in :code:`ivy/functional/backends/backend_name/creation.py`
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

This makes it clear that these backend-specific functions are only enterred into once the correct :code:`device`
has been determined.

However, the :code:`device` argument for functions without the :code:`@infer_device` decorator
is **not** handled by `infer_device`_,
and so these defaults must be handled by the backend-specific implementations themselves,
by calling :code:`ivy.default_device` internally.

**Round Up**

This should have hopefully given you a good feel for devices, and how these are handled in Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `devices discussion`_,
or reach out on `discord`_ in the `devices channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/-Y1Ofk72TLY" class="video">
    </iframe>