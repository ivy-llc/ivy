Docstrings
==========

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`spec/API_specification/array_api`: https://github.com/data-apis/array-api/tree/main/spec/API_specification/array_api

As we did when explaining :ref:`Type Hints`, we will again use the functions :code:`ivy.tan`, :code:`ivy.roll` and
:code:`ivy.add` as exemplars.

Firstly, if the function exists in the `Array API Standard`_, the we start with the corresponding docstring as a
template. These docstrings can be found under `spec/API_specification/array_api`_.

The `Array API Standard`_ docstring for :code:`tan` is as follows:

.. code-block:: python

    Calculates an implementation-dependent approximation to the tangent, having domain ``(-infinity, +infinity)`` and codomain ``(-infinity, +infinity)``, for each element ``x_i`` of the input array ``x``. Each element ``x_i`` is assumed to be expressed in radians.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

    Parameters
    ----------
    x: array
        input array whose elements are expressed in radians. Should have a floating-point data type.

    Returns
    -------
    out: array
        an array containing the tangent of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

This is a good starting point. But we need to make some changes. Firstly, given that we are using type hints, repeating
all of the types also in the docs would be a needless duplication. Therefore, we simply remove all type info from the
docstring like so:

.. code-block:: diff

    -x: array
    +x
    -out: array
    +out

The `Array API Standard`_ defines a subset of behaviour that each function must adhere to.
Ivy extends many of these functions with additional behaviour and arguments.
In the case of :code:`ivy.tan`, there is also the argument :code:`out` which needs to be added to the docstring,
like so:

.. code-block:: diff

    +out
    +    optional output, for writing the result to. It must have a shape that the inputs
    +    broadcast to.

Because of this :code:`out` argument in the input, we also need to rename the :code:`out` argument in the return, which
is the default name used in the `Array API Standard`_. We change this to :code:`ret`:

.. code-block:: diff

    -out
    +ret

Finally, we add a section in the docstring which explains that it has been modified from the version available in the
`Array API Standard`_:

.. code-block:: diff

    +This method conforms to the `Array API Standard
    +<https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    +`docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.tan.html>`_
    +in the standard. The descriptions above assume an array input for simplicity, but
    +the method also accepts :code:`ivy.Container` instances in place of
    +:code:`ivy.Array` or :code:`ivy.NativeArray` instances, as shown in the type hints
    +and also the examples below.

Following these changes, the new docstring is as follows:

.. code-block:: python

    Calculates an implementation-dependent approximation to the tangent, having
    domain ``(-infinity, +infinity)`` and codomain ``(-infinity, +infinity)``, for each
    element ``x_i`` of the input array ``x``. Each element ``x_i`` is assumed to be
    expressed in radians.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

    Parameters
    ----------
    x
        input array whose elements are expressed in radians. Should have a
        floating-point data type.
    out
        optional output, for writing the result to. It must have a shape that the inputs
        broadcast to.

    Returns
    -------
    ret
        an array containing the tangent of each element in ``x``. The return must have a
        floating-point data type determined by :ref:`type-promotion`.

    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.tan.html>`_
    in the standard. The descriptions above assume an array input for simplicity, but
    the method also accepts :code:`ivy.Container` instances in place of
    :code:`ivy.Array` or :code:`ivy.NativeArray` instances, as shown in the type hints
    and also the examples below.