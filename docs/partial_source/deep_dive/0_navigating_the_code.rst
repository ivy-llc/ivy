Navigating the Code
===================

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`project structure discussion`: https://github.com/unifyai/ivy/discussions/1311
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`project structure channel`: https://discord.com/channels/799879767196958751/982737793476345888
.. _`Array API Standard convention`: https://data-apis.org/array-api/2021.12/API_specification/array_object.html#api-specification-array-object--page-root
.. _`flake8`: https://flake8.pycqa.org/en/latest/index.html
.. _`pre-commit guide`: https://lets-unify.ai/ivy/contributing/0_setting_up.html#pre-commit

Categorization
--------------

Ivy uses the following categories taken from the `Array API Standard`_:

* `constants <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/constants.py>`_
* `creation <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/creation.py>`_
* `data_type <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/data_type.py>`_
* `elementwise <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/elementwise.py>`_
* `linear_algebra <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/linear_algebra.py>`_
* `manipulation <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/manipulation.py>`_
* `searching <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/searching.py>`_
* `set <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/set.py>`_
* `sorting <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/sorting.py>`_
* `statistical <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/statistical.py>`_
* `utility <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/utility.py>`_

In addition to these, we also add the following categories,
used for additional functions in Ivy that are not in the `Array API Standard`_:

* `activations <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/activations.py>`_
* `compilation <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/compilation.py>`_
* `device <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/device.py>`_
* `general <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/general.py>`_
* `gradients <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/gradients.py>`_
* `image <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/image.py>`_
* `layers <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/layers.py>`_
* `losses <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/losses.py>`_
* `meta <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/meta.py>`_
* `nest <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/nest.py>`_
* `norms <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/norms.py>`_
* `random <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/random.py>`_

Some functions that you're considering adding might overlap several of these categorizations,
and in such cases you should look at the other functions included in each file,
and use your best judgement for which categorization is most suitable.

We can always suggest a more suitable location when reviewing your pull request if needed 🙂

Submodule Design
----------------

Ivy is designed so that all methods are called directly from the :code:`ivy` namespace, such as :code:`ivy.matmul`,
and not :code:`ivy.some_namespace.matmul`. Therefore, inside any of the folders :code:`ivy.functional.ivy`,
:code:`ivy.functional.backends.some_backend`, :code:`ivy.functional.backends.another_backend` the functions can be moved
to different files or folders without breaking anything at all. This makes it very simple to refactor and re-organize
parts of the code structure in an ongoing manner.

The :code:`__init__.py` inside each of the subfolders are very similar,
importing each function via :code:`from .file_name import *`
and also importing each file as a submodule via :code:`from . import file_name`.
For example, an extract from
`ivy/ivy/functional/ivy/__init__.py <https://github.com/unifyai/ivy/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/ivy/functional/ivy/__init__.py>`_
is given below:

.. code-block:: python

    from . import elementwise
    from .elementwise import *
    from . import general
    from .general import *
    # etc.


Ivy API
-------

All function signatures for the Ivy API are defined in the :code:`ivy.functional.ivy` submodule. Functions written here
look something like the following, (explained in much more detail in the following sections):


.. code-block:: python


    def my_func(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        axes: Union[int, Tuple[int], List[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        My function does something cool.

        .. note::
            This is an important note.

        **Special Cases**

        For this particular case,

        - If ``x`` is ``NaN``, do something
        - If ``y`` is ``-0``, do something else
        - etc.

        Parameters
        ----------
        x
            input array. Should have a numeric data type.
        axes
            the axes along which to perform the op.
        dtype
            array data type.
        device
            the device on which to place the new array.
        out
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a cooler array.

        Examples
        --------

        Some cool examples go here
        """
        return ivy.current_backend(x).my_func(x, axes, dtype=dtype, device=device, out=out)

We follow the `Array API Standard convention`_ about positional and keyword arguments.

* Positional parameters must be positional-only parameters. Positional-only parameters have no externally-usable name. When a method accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order.
* Optional parameters must be keyword-only arguments

This convention makes it easier for us to modify functions in the future. Keyword-only parameters will mandate the use
of argument names when calling functions, and this will increase our flexibility for extending function behaviour in
future releases without breaking forward compatibility. Similar arguments can be kept together in the argument list,
rather than us needing to add these at the very end to ensure positional argument behaviour remains the same.

The :code:`dtype`, :code:`device` and :code:`out` arguments are always keyword-only.
Arrays always have type hint :code:`Union[ivy.Array, ivy.NativeArray]` in the input and :code:`ivy.Array` in the output.
All functions which produce a single array include the :code:`out` argument.
The reasons for each of these features are explained in the following sections.

Backend API
-----------

Code in the backend submodules such as :code:`ivy.functional.backends.torch` should then look something like:

.. code-block:: python


    def my_func(
        x: torch.Tensor,
        /,
        axes: Union[int, Tuple[int], List[int]],
        *,
        dtype: torch.dtype,
        device: torch.device,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.something_cool(x, axes, dtype, device, out)

The :code:`dtype`, :code:`device` and :code:`out` arguments are again all keyword-only,
but :code:`dtype` and :code:`device` are now required arguments, rather than optional as they were in the Ivy API.
All arrays also now have the same type hint :code:`torch.Tensor`,
rather than :code:`Union[ivy.Array, ivy.NativeArray]` in the input and :code:`ivy.Array` in the output.
The backend methods also should not add a docstring.
Again, the reasons for these features are explained in the following sections.

Submodule Helper Functions
--------------------------

At times, helper functions specific to submodule is required to:

* keep the code clean and readable
* be imported in their respective backend implementations

To have a better idea on this, let's look at an example!

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
        /,
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
        /,
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
        /,
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
        /,
        fill_value: Union[int, float],
        *,
        dtype: torch.dtype,
        device: torch.device,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
        return torch.full_like(x, fill_value, dtype=dtype, device=device)

**Round Up**

This should have hopefully given you a good feel for how to navigate the Ivy codebase.

If you're ever unsure of how best to proceed,
please feel free to engage with the `project structure discussion`_,
or reach out on `discord`_ in the `project structure channel`_!


**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/67UYuLcAKbY" class="video">
    </iframe>
