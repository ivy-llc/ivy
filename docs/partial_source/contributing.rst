Contributing to Ivy
===================

.. _`Array API`: https://data-apis.org/array-api/latest/


Submodule Design
----------------

Many already-implemented methods will need to be moved into new locations during various stages of refactoring.
The package is designed so all methods are called directly from the :code:`ivy` namespace, such as :code:`ivy.matmul`,
and not :code:`ivy.some_namespace.matmul`. Therefore, inside any of the folders :code:`ivy.functional.ivy`,
:code:`ivy.functional.backends.some_backend`, :code:`ivy.functional.backends.another_backend` the functions can be moved
to different files or folders without breaking anything. This makes it very simple to continually refactor and re-organize
the code structure in an ongoing manner.

Currently, we are in the process of refactoring things to more closely follow the Array API standard. Many methods will
need to be moved to new locations. Again, this is not a problem, provided the :code:`__init__` files have the correct
imports. Generally, these files look like the following, so that both the submodule namespace is imported but also
all methods.

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

        :param x: input array.
        :param axes: the axes along which to perform the op.
        :param dtype: array data type.
        :param dev: the device on which to place the new array.
        :return: a cooler array.
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

All functions which adhere to the `Array API`_ standard should be placed in the submodule :code:`ivy.functional.ivy.array_api`,
and should also be placed in the correct file in alignment with the categories used in the standard.


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

All backend functions which adhere to the `Array API`_ standard should also be placed in submodules such as
:code:`ivy.functional.backends.torch.array_api`, and should also be placed in the correct file in alignment with the
categories used in the standard.


Array Operators
---------------

Array operators are defined in the :code:`ivy.array` submodule. Operators written here should adhere to the following format:

.. code-block:: python


    @_native_wrapper
    def __pow__(self, power):
        return ivy.builtin_pow(self, power)

There is no need to write docstrings or type hints for these methods, as they should always defer to a method such as
:code:`ivy.builtin_some_op`, which will itself have a docstring and type hints.
The remaining code is essentially simple wrapper code around this builtin ivy method.

The associated ivy backend methods should be placed in the same file as the operators. For example, :code:`__pow__` is
an arithmetic operator, and so this operator should be placed in the submodule :code:`ivy.array.array_api.arithmetic_operators`.
The method :code:`ivy.builtin_pow` should also be placed in :code:`ivy.array.array_api.arithmetic_operators`.

For most methods and backends these are very simple to implement, such as :code:`ivy.builtin_pow` below:

.. code-block:: python

    # noinspection PyShadowingBuiltins
    def builtin_pow(self: ivy.Array,
                    other: Union[int, float, ivy.Array]) \
            -> ivy.Array:
        """
        Calculates an implementation-dependent approximation of exponentiation by raising each element (the base) of an
        array instance to the power of other_i (the exponent), where other_i is the corresponding element of the array other.

        :param self: array instance whose elements correspond to the exponentiation base. Should have a numeric data type.
        :param other: other array whose elements correspond to the exponentiation exponent. Must be compatible with x
                        (see Broadcasting). Should have a numeric data type.
        :return: an array containing the element-wise results. The returned array must have a data type determined by
                  Type Promotion Rules.
        """
        return self.__pow__(other)

However, for some backends this does not work. For example, MXNet does not support reshaping arrays to 0-dim arrays,
but this is required by the standard. Therefore, we've written custom methods for handling 0-dim arrays. For backends
such as this where more customization is needed, then we must simply redefine these methods, such as :code:`ivy.builtin_pow`,
in the associated backend submodule, in this case :code:`ivy.functional.backends.mxnet.array_builtins.array_api.arithmetic_operators`.

The custom MXNet code is as follows, with the addition of an MXNet-specific function decorator to properly handle flat arrays:

.. code-block:: python

    @_handle_flat_arrays_in_out
    def builtin_pow(self: mx.ndarray.ndarray.NDArray,
                    other: Union[int, float, mx.ndarray.ndarray.NDArray]) \
                -> mx.ndarray.ndarray.NDArray:
        return self.__pow__(other)

Again, a docstring is not needed given that this is the same as the one provided in :code:`ivy.array.array_api.arithmetic_operators`.
For other backends, we do not need to specify a custom :code:`builtin_pow` method. These will default to the version implemented in
:code:`ivy.array.array_api.arithmetic_operators` if no custom implementation is provided.