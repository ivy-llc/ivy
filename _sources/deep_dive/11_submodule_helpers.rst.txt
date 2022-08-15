Submodule Helper Functions
==========================

.. _`flake8`: https://flake8.pycqa.org/en/latest/index.html
.. _`pre-commit guide`: https://lets-unify.ai/ivy/contributing/0_setting_up.html#pre-commit

At times, helper functions specific to submodule is required to:

* keep the code clean and readable
* be imported in their respective backend implementations

To have a better idea on this, let's look at some examples!

Example 1
---------

**Helper in Ivy**

.. code-block:: python

    # in ivy/functional/ivy/creation.py
    def _assert_fill_value_and_dtype_are_compatible(dtype, fill_value):
        assert (
            (ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype))
            and isinstance(fill_value, int)
        ) or (
            ivy.is_float_dtype(dtype)
            and isinstance(fill_value, float)
            or (isinstance(fill_value, bool))
        ), "the fill_value and data type are not compatible"

In the :code:`full_like` function in :code:`creation.py`, the types of
:code:`fill_value` and :code:`dtype` has to be verified to avoid errors. This
check has to be applied to all backends, which means the related code is common
and identical. In this case, we can extract the code to be a helper function on
its own, placed in its related submodule (:code:`creation.py` here). In this
example, the helper function is named as
:code:`_assert_fill_value_and_dtype_are_compatible`.

Then, we import this submodule-specific helper function to the respective backends,
where examples for each backend is shown below.

**Jax**

.. code-block:: python

    # in ivy/functional/backends/jax/creation.py
    from ivy.functional.ivy.creation import _assert_fill_value_and_dtype_are_compatible

    def full_like(
        x: JaxArray,
        fill_value: Union[int, float],
        *,
        dtype: jnp.dtype,
        device: jaxlib.xla_extension.Device,
        out: Optional[JaxArray] = None
    ) -> JaxArray:
        _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
        return _to_device(
            jnp.full_like(x, fill_value, dtype=dtype),
            device=device,
        )

**NumPy**

.. code-block:: python

    # in ivy/functional/backends/numpy/creation.py
    from ivy.functional.ivy.creation import _assert_fill_value_and_dtype_are_compatible

    def full_like(
        x: np.ndarray,
        fill_value: Union[int, float],
        *,
        dtype: np.dtype,
        device: str,
        out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
        return _to_device(np.full_like(x, fill_value, dtype=dtype), device=device)

**TensorFlow**

.. code-block:: python

    # in ivy/functional/backends/tensorflow/creation.py
    from ivy.functional.ivy.creation import _assert_fill_value_and_dtype_are_compatible

    def full_like(
        x: Union[tf.Tensor, tf.Variable],
        fill_value: Union[int, float],
        *,
        dtype: tf.DType,
        device: str,
        out: Union[tf.Tensor, tf.Variable] = None
    ) -> Union[tf.Tensor, tf.Variable]:
        _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
        with tf.device(device):
            return tf.experimental.numpy.full_like(x, fill_value, dtype=dtype)

**Torch**

.. code-block:: python

    # in ivy/functional/backends/torch/creation.py
    from ivy.functional.ivy.creation import _assert_fill_value_and_dtype_are_compatible

    def full_like(
        x: torch.Tensor,
        fill_value: Union[int, float],
        *,
        dtype: torch.dtype,
        device: torch.device,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
        return torch.full_like(x, fill_value, dtype=dtype, device=device)

Example 2
---------

**Helper in Ivy**

.. code-block:: python

    # in ivy/functional/ivy/data_type.py
    def _is_valid_dtypes_attributes(fn: Callable) -> bool:
        if hasattr(fn, "supported_dtypes") and hasattr(fn, "unsupported_dtypes"):
            fn_supported_dtypes = fn.supported_dtypes
            fn_unsupported_dtypes = fn.unsupported_dtypes
            if isinstance(fn_supported_dtypes, dict):
                if isinstance(fn_unsupported_dtypes, dict):
                    backend_str = ivy.current_backend_str()
                    if (
                        backend_str in fn_supported_dtypes
                        and backend_str in fn_unsupported_dtypes
                    ):
                        return False
            else:
                if isinstance(fn_unsupported_dtypes, tuple):
                    return False
        return True

In the :code:`function_supported_dtypes` and :code:`function_unsupported_dtypes`
functions in :code:`data_type.py`, we have to ensure that the attributes
:code:`supported_dtypes` and :code:`unsupported_dtypes` do not exist for the
same backend. However, both of the functions only exist in the Ivy
:code:`data_type.py` submodule without backend implementations. Therefore, the
purpose of creating the helper function - :code:`_is_valid_dtypes_attributes`
in this case is to keep code clean and readable.

As the functions and helper exist in the same submodule, we can directly use
the helper function without importing.

**function_supported_dtypes**

.. code-block:: python

    # in ivy/functional/backends/jax/creation.py
    from ivy.functional.ivy.creation import _assert_fill_value_and_dtype_are_compatible

    def function_supported_dtypes(fn: Callable) -> Tuple:
        if not _is_valid_dtypes_attributes(fn):
            raise Exception(
                "supported_dtypes and unsupported_dtypes attributes cannot both \
                 exist in a particular backend"
            )
        supported_dtypes = tuple()
        if hasattr(fn, "supported_dtypes"):
            fn_supported_dtypes = fn.supported_dtypes
            if isinstance(fn_supported_dtypes, dict):
                backend_str = ivy.current_backend_str()
                if backend_str in fn_supported_dtypes:
                    supported_dtypes += fn_supported_dtypes[backend_str]
                if "all" in fn_supported_dtypes:
                    supported_dtypes += fn_supported_dtypes["all"]
            else:
                supported_dtypes += fn_supported_dtypes
        return tuple(set(supported_dtypes))

**function_unsupported_dtypes**

.. code-block:: python

    # in ivy/functional/backends/numpy/creation.py
    from ivy.functional.ivy.creation import _assert_fill_value_and_dtype_are_compatible

    def function_unsupported_dtypes(fn: Callable) -> Tuple:
        if not _is_valid_dtypes_attributes(fn):
            raise Exception(
                "supported_dtypes and unsupported_dtypes attributes cannot both \
                 exist in a particular backend"
            )
        unsupported_dtypes = ivy.invalid_dtypes
        if hasattr(fn, "unsupported_dtypes"):
            fn_unsupported_dtypes = fn.unsupported_dtypes
            if isinstance(fn_unsupported_dtypes, dict):
                backend_str = ivy.current_backend_str()
                if backend_str in fn_unsupported_dtypes:
                    unsupported_dtypes += fn_unsupported_dtypes[backend_str]
                if "all" in fn_unsupported_dtypes:
                    unsupported_dtypes += fn_unsupported_dtypes["all"]
            else:
                unsupported_dtypes += fn_unsupported_dtypes
        return tuple(set(unsupported_dtypes))
