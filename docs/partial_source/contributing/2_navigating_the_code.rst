Navigating the Code
===================

.. _`Array API`: https://data-apis.org/array-api/latest/

Submodule Design
----------------

Ivy is designed so that all methods are called directly from the :code:`ivy` namespace, such as :code:`ivy.matmul`,
and not :code:`ivy.some_namespace.matmul`. Therefore, inside any of the folders :code:`ivy.functional.ivy`,
:code:`ivy.functional.backends.some_backend`, :code:`ivy.functional.backends.another_backend` the functions can be moved
to different files or folders without breaking anything at all. This makes it very simple to refactor and re-organize
parts of the code structure in an ongoing manner.

.. code-block:: python

    from . import dtype
    from .dtype import *
    from . import general
    from .general import *
    # etc.


Ivy API
-------

All function signatures for the Ivy API are defined in the :code:`ivy.functional.ivy` submodule. Functions written here
look something like the following, (explained in much more detail in the following sections):


.. code-block:: python


    def my_func(x: Union[ivy.Array, ivy.NativeArray],
                axes: Union[int, Tuple[int], List[int]],
                *,
                dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
                device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
                out: Optional[ivy.Array] = None) \
            -> ivy.Array:
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
        ret:
            a cooler array.

        Examples
        --------

        Some cool examples go here
        """
        return _cur_framework(x).my_func(x, axes, dtype=dtype, device=device, out=out)

The :code:`dtype`, :code:`device` and :code:`out` arguments are always keyword-only.
Arrays always have type hint :code:`Union[ivy.Array, ivy.NativeArray]` in the input and :code:`ivy.Array` in the output.
All functions which produce a single array include the :code:`out` argument.
The reasons for these features are explained in the following sections.

Backend API
-----------

Code in the backend submodules such as :code:`ivy.functional.backends.torch` should then look something like:

.. code-block:: python


    def my_func(x: torch.Tensor,
                axes: Union[int, Tuple[int], List[int]],
                *,
                dtype: torch.dtype,
                device: torch.device,
                out: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        return torch.something_cool(x, dtype, device, out)

The :code:`dtype`, :code:`device` and :code:`out` arguments are again all keyword-only,
but :code:`dtype` and :code:`device` and now required, rather than optional as they were in the Ivy API.
All arrays also now have the same type hint :code:`torch.Tensor`,
rather than :code:`Union[ivy.Array, ivy.NativeArray]` in the input and :code:`ivy.Array` in the output.
The backend methods also should not add a docstring.
Again, the reasons for these features are explained in the following sections.