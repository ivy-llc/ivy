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
.. _`ivy.set_soft_device_mode`: https://github.com/unifyai/ivy/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/functional/ivy/device.py#L292
.. _`@handle_device_shifting`: https://github.com/unifyai/ivy/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/func_wrapper.py#L797
.. _`ivy.functional.ivy`: https://github.com/unifyai/ivy/tree/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/functional/ivy
.. _`tensorflow soft device handling function`: https://github.com/unifyai/ivy/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/functional/backends/tensorflow/device.py#L102
.. _`numpy soft device handling function`: https://github.com/unifyai/ivy/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/functional/backends/numpy/device.py#L88
.. _`ivy implementation`: https://github.com/unifyai/ivy/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/functional/ivy/device.py#L138
.. _`tf.device`: https://www.tensorflow.org/api_docs/python/tf/device
.. _`ivy.DefaultDevice`: https://github.com/unifyai/ivy/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/functional/ivy/device.py#L52
.. _`__enter__`: https://github.com/unifyai/ivy/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/functional/ivy/device.py#L76
.. _`__exit__`: https://github.com/unifyai/ivy/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/ivy/functional/ivy/device.py#L98
.. _`ivy.unset_soft_device_mode()`: https://github.com/unifyai/ivy/blob/2f90ce7b6a4c8ddb7227348d58363cd2a3968602/ivy/functional/ivy/device.py#L317
.. _`ivy.unset_default_device()`: https://github.com/unifyai/ivy/blob/2f90ce7b6a4c8ddb7227348d58363cd2a3968602/ivy/functional/ivy/device.py#L869
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`devices channel`: https://discord.com/channels/799879767196958751/982738108166602752

The devices currently supported by Ivy are as follows:

* cpu
* gpu:idx
* tpu:idx

In a similar manner to the :class:`ivy.Dtype` and :class:`ivy.NativeDtype` classes (see :ref:`Data Types`), there is both an `ivy.Device`_ class and an :class:`ivy.NativeDevice` class, with :class:`ivy.NativeDevice` initially set as an `empty class`_.
The :class:`ivy.Device` class derives from :code:`str`, and has simple logic in the constructor to verify that the string formatting is correct.
When a backend is set, the :class:`ivy.NativeDevice` is replaced with the backend-specific `device class`_.

Device Module
-------------

The `device.py`_ module provides a variety of functions for working with devices.
A few examples include :func:`ivy.get_all_ivy_arrays_on_dev` which gets all arrays which are currently alive on the specified device, :func:`ivy.dev` which gets the device for input array, and :func:`ivy.num_gpus` which determines the number of available GPUs for use with the backend framework.

Many functions in the :mod:`device.py` module are *convenience* functions, which means that they do not directly modify arrays, as explained in the :ref:`Function Types` section.

For example, the following are all convenience functions: `ivy.total_mem_on_dev`_, which gets the total amount of memory for a given device, `ivy.dev_util`_, which gets the current utilization (%) for a given device, `ivy.num_cpu_cores`_, which determines the number of cores available in the CPU, and `ivy.default_device`_, which returns the correct device to use.

`ivy.default_device`_ is arguably the most important function.
Any function in the functional API that receives a :code:`device` argument will make use of this function, as explained below.

Arguments in other Functions
----------------------------

Like with :code:`dtype`, all :code:`device` arguments are also keyword-only.
All creation functions include the :code:`device` argument, for specifying the device on which to place the created array.
Some other functions outside of the :code:`creation.py` submodule also support the :code:`device` argument, such as :func:`ivy.random_uniform` which is located in :mod:`random.py`, but this is simply because of dual categorization.
:func:`ivy.random_uniform` is also essentially a creation function, despite not being located in :mod:`creation.py`.

The :code:`device` argument is generally not included for functions which accept arrays in the input and perform operations on these arrays.
In such cases, the device of the output arrays is the same as the device for the input arrays.
In cases where the input arrays are located on different devices, an error will generally be thrown, unless the function is specific to distributed training.

The :code:`device` argument is handled in `infer_device`_ for all functions which have the :code:`@infer_device` decorator, similar to how :code:`dtype` is handled.
This function calls `ivy.default_device`_ in order to determine the correct device.
As discussed in the :ref:`Function Wrapping` section, this is applied to all applicable functions dynamically during `backend setting`_.

Overall, `ivy.default_device`_ infers the device as follows:

#. if the :code:`device` argument is provided, use this directly
#. otherwise, if an array is present in the arguments (very rare if the :code:`device` argument is present), set :code:`arr` to this array.
   This will then be used to infer the device by calling :func:`ivy.dev` on the array
#. otherwise, if no arrays are present in the arguments (by far the most common case if the :code:`device` argument is present), then use the global default device, which currently can either be :code:`cpu`, :code:`gpu:idx` or :code:`tpu:idx`.
   The default device is settable via :func:`ivy.set_default_device`.

For the majority of functions which defer to `infer_device`_ for handling the device, these steps will have been followed and the :code:`device` argument will be populated with the correct value before the backend-specific implementation is even entered into.
Therefore, whereas the :code:`device` argument is listed as optional in the ivy API at :mod:`ivy/functional/ivy/category_name.py`, the argument is listed as required in the backend-specific implementations at :mod:`ivy/functional/backends/backend_name/category_name.py`.

This is exactly the same as with the :code:`dtype` argument, as explained in the :ref:`Data Types` section.

Let's take a look at the function :func:`ivy.zeros` as an example.

The implementation in :mod:`ivy/functional/ivy/creation.py` has the following signature:

.. code-block:: python

    @outputs_to_ivy_arrays
    @handle_out_argument
    @infer_dtype
    @infer_device
    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Array:

Whereas the backend-specific implementations in :mod:`ivy/functional/backends/backend_name/creation.py` all list :code:`device` as required.

Jax:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: jnp.dtype,
        device: jaxlib.xla_extension.Device,
    ) -> JaxArray:

NumPy:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: np.dtype,
        device: str,
    ) -> np.ndarray:

TensorFlow:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: tf.DType,
        device: str,
    ) -> Tensor:

PyTorch:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:

This makes it clear that these backend-specific functions are only enterred into once the correct :code:`device` has been determined.

However, the :code:`device` argument for functions without the :code:`@infer_device` decorator is **not** handled by `infer_device`_, and so these defaults must be handled by the backend-specific implementations themselves, by calling :func:`ivy.default_device` internally.

Device handling
---------------

Different frameworks handle devices differently while performing an operation. For example, torch expects
all the tensors to be on the same device while performing an operation, or else, it throws a device exception. On the other hand, tensorflow
doesn't care about this, it moves all the tensors to the same device before performing an operation.

**Controlling Device Handling Behaviour**

In Ivy, users can control the device on which the operation is to be executed using `ivy.set_soft_device_mode`_ flag. There are two cases for this, 
either the soft device mode is set to :code:`True` or :code:`False`.

1. When :code:`ivy.set_soft_device_mode(True)`:

a. All the input arrays are moved to :code:`ivy.default_device()` while performing an operation. If the array is already present
in the default device, no device shifting is done.

In the example below, even though the input arrays :code:`x` and :code:`y` are created on different devices('cpu' and 'gpu:0'), the arrays
are moved to :code:`ivy.default_device()` while performing :code:`ivy.add` operation, and the output array will be on this device.

.. code-block:: python
    
    ivy.set_backend("torch")
    ivy.set_soft_device_mode(True)
    x = ivy.array([1], device="cpu")
    y = ivy.array([34], device="gpu:0")
    ivy.add(x, y)

2. When :code:`ivy.set_soft_device_mode(False)`:

a. If any of the input arrays are on a different device, a device exception is raised.

In the example below, since the input arrays are on different devices('cpu' and 'gpu:0'), an :code:`IvyBackendException` is raised while performing :code:`ivy.add`.

.. code-block:: python

    ivy.set_backend("torch")
    ivy.set_soft_device_mode(False)
    x = ivy.array([1], device="cpu")
    y = ivy.array([34], device="gpu:0")
    ivy.add(x, y)

This is the exception you will get while running the code above:

.. code-block:: python

    IvyBackendException: torch: add:   File "/content/ivy/ivy/utils/exceptions.py", line 210, in _handle_exceptions
        return fn(*args, **kwargs)
    File "/content/ivy/ivy/func_wrapper.py", line 1013, in _handle_nestable
        return fn(*args, **kwargs)
    File "/content/ivy/ivy/func_wrapper.py", line 905, in _handle_out_argument
        return fn(*args, out=out, **kwargs)
    File "/content/ivy/ivy/func_wrapper.py", line 441, in _inputs_to_native_arrays
        return fn(*new_args, **new_kwargs)
    File "/content/ivy/ivy/func_wrapper.py", line 547, in _outputs_to_ivy_arrays
        ret = fn(*args, **kwargs)
    File "/content/ivy/ivy/func_wrapper.py", line 358, in _handle_array_function
        return fn(*args, **kwargs)
    File "/content/ivy/ivy/func_wrapper.py", line 863, in _handle_device_shifting
        raise ivy.utils.exceptions.IvyException(
    During the handling of the above exception, another exception occurred:
    Expected all input arrays to be on the same device, but found atleast two devices - ('cpu', 'gpu:0'), 
    set `ivy.set_soft_device_mode(True)` to handle this problem.

b. If all the input arrays are on the same device, the operation is executed without raising any device exceptions.

The example below runs without issues since both the input arrays are on 'gpu:0' device:

.. code-block:: python

    ivy.set_backend("torch")
    ivy.set_soft_device_mode(False)
    x = ivy.array([1], device="gpu:0")
    y = ivy.array([34], device="gpu:0")
    ivy.add(x, y)

The code to handle all these cases are present inside `@handle_device_shifting`_ decorator, which is wrapped around
all the functions that accept at least one array as input(except mixed and compositional functions) in `ivy.functional.ivy`_ submodule. The decorator calls
:code:`ivy.handle_soft_device_variable` function under the hood to handle device shifting for each backend.

**Soft Device Handling Function**

There is a backend specific implementation of :code:`ivy.handle_soft_device_variable` function for numpy and tensorflow. The reason being, for numpy there 
is no need for device shifting as it only support 'cpu' device, whereas, tensorflow automatically moves the inputs to 'gpu' if one is available and there is no way to turn this
off globally.

The `numpy soft device handling function`_ just returns the inputs of the operation as it is without making any changes.
Whereas the `tensorflow soft device handling function`_ move the input arrays to :code:`ivy.default_device()` using 
`tf.device`_ context manager.

For the rest of the frameworks, the `ivy implementation`_ of soft device handling function is used, which loops through
the inputs of the function and move the arrays to :code:`ivy.default_device()`, if not already on that device.

**Forcing Operations on User Specified Device**

The `ivy.DefaultDevice`_ context manager can be used to force the operations to be performed on to a specific device. For example,
in the code below, both :code:`x` and :code:`y` will be moved from 'gpu:0' to 'cpu' device and :code:`ivy.add` operation will be performed on 'cpu' device:

.. code-block:: python

    x = ivy.array([1], device="gpu:0")
    y = ivy.array([34], device="gpu:0")
    with ivy.DefaultDevice("cpu"):
        z = ivy.add(x, y)

On entering :code:`ivy.DefaultDevice("cpu")` context manager, under the hood, the default device is set to 'cpu' and soft device
mode is turned on. All these happens under the `__enter__`_ method of the
context manager. So from now on, all the operations will be executed on 'cpu' device.

On exiting the context manager(`__exit__`_ method), the default device and soft device mode is reset to the previous state using `ivy.unset_default_device()`_ and
`ivy.unset_soft_device_mode()`_ respectively, to move back to the previous state.

**Round Up**

This should have hopefully given you a good feel for devices, and how these are handled in Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `devices channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/-Y1Ofk72TLY" class="video">
    </iframe>
