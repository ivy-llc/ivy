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
should adhere to the following type hint format:


.. code-block:: python


    def my_func(x: Union[ivy.Array, ivy.NativeArray],
                axes: Union[int, Tuple[int], List[int]],
                dtype: Optional[Union[ivy.Dtype, str]] = None,
                dev: Optional[Union[ivy.Dev, str]] = None) \
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
        x:
            input array. Should have a numeric data type.
        axes:
            the axes along which to perform the op.
        dtype:
            array data type.
        dev:
            the device on which to place the new array.

        Returns
        -------
        out:
            a cooler array.
        """
        return _cur_framework(x).my_func(x, dtype, dev)

Note that the input array has type :code:`Union[ivy.Array, ivy.NativeArray]` whereas the output array has type
:code:`ivy.Array`. This is the case for all functions in the ivy API.
We always return an :code:`ivy.Array` instance to ensure that any subsequent Ivy code is fully framework-agnostic, with
all operators performed on the array being handled by Ivy, and not the backend framework. However, there is no need to
prevent native arrays from being permitted in the input. For Ivy methods which wrap backend-specific implementations, the
input would need to be converted to a native array (such as :code:`torch.Tensor`) anyway before calling the backend method,
and for Ivy methods implemented as a composition of other Ivy methods such as :code:`ivy.lstm_update`, the native inputs can
just be converted to :code:`ivy.Array` instances before executing the Ivy implementation.

As for the :code:`axes` arg, generally the `Array API`_ standard dictates that shapes, axes and other similar args should be
of type :code:`Tuple[int]` when representing a sequence, not :code:`List[int]`. However, in order to make Ivy code
less brittle, we accept both tuples and lists for such arguments. This does not break the standard, as the standard is only
intended to define a subset of required function behaviour. The standard can be freely extended, as we are doing here.

As for the other arguments in the example above, :code:`dtype` and :code:`dev` do not need to be added to all methods,
these are just examples. These should be added to all creation methods though. Note that for both of these, the type is a
:code:`Union` including :code:`str`. This is because, in order to remain fully framework agnostic, Ivy accepts string
representations of devices and data types, such as :code:`"int32"`, :code:`"float32"`, :code:`"bool"`, :code:`"cpu"`,
:code:`"gpu0"`, :code:`"gpu2"` etc.

All functions which adhere to the `Array API`_ standard should be placed in the correct file in alignment with the
categories used in the standard.

Backend API
-----------

Code in the backend submodules such as :code:`ivy.functional.backends.torch` should then look something like:

.. code-block:: python


    def my_func(x: torch.Tensor,
                dtype: Optional[Union[torch.dtype, str]] = None,
                dev: Optional[Union[torch.device, str]] = None) \
            -> torch.Tensor:
        dtype = ivy.dtype_from_str(ivy.default_dtype(dtype, x))
        dev = ivy.dev_from_str(ivy.default_dev(dev, x))
        return torch.something_cool(x, dtype, dev)

Specifically, we should use type hints for all arguments in the Ivy API and also the backend APIs. These type hints
should be identical apart from all :code:`ivy.Array`, :code:`ivy.Dtype` and :code:`ivy.Dev` types replaced by
framework-specific types, in this case :code:`torch.Tensor`, :code:`torch.dtype` and :code:`torch.device`.

The backend methods should not add a docstring, as this would be identical to the docstring provided in the Ivy API.

All backend functions which adhere to the `Array API`_ standard should also be placed in the correct file in alignment with the
categories used in the standard.