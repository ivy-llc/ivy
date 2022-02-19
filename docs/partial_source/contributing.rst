Contributing to Ivy
===================

Ivy API
-------

All function signatures for the Ivy API are defined in the :code:`ivy.functional.ivy` submodule. Functions written here
should adhere to the following format:


.. code-block:: python


    def my_func(x: ivy.Array,
                dtype: Optional[Union[ivy.Dtype, str]] = None,
                dev: Optional[Union[ivy.Dev, str]] = None):
        """
        My function does something cool.

        :param x: input array.
        :param dtype: array data type.
        :param dev: the device on which to place the new array.
        :return: a cooler array.
        """
        return _cur_framework(x).my_func(x, dtype, dev)

Backend API
-----------

Code in the backend submodules such as :code:`ivy.functional.backends.torch` should then look something like:

.. code-block:: python


    def my_func(x: torch.Tensor,
                dtype: Optional[Union[torch.dtype, str]] = None,
                dev: Optional[Union[torch.device, str]] = None):
        dtype = ivy.dtype_from_str(ivy.default_dtype(dtype, x))
        dev = ivy.dev_from_str(ivy.default_dev(dev, x))
        return torch.something_cool(x, dtype, dev)

Specifically, we should use type hints for all arguments in the Ivy API and also the backend APIs. These type hints
should be identical apart from all :code:`ivy.Array`, :code:`ivy.Dtype` and :code:`ivy.Dev` types replaced by
framework-specific types.

The backend methods should not add a docstring, as this would be identical to the docstring provided in the Ivy API.
