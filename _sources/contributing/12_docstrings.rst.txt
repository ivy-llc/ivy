Docstrings
==========

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`spec/API_specification/array_api`: https://github.com/data-apis/array-api/tree/main/spec/API_specification/array_api

All functions in the Ivy API at :code:`ivy/functional/ivy/category_name.py` should have full and thorough docstrings.
In contrast, all backend implementations at
:code:`ivy/functional/backends/backend_name/category_name.py` should not have any docstrings,
on account that these are effectively just different instantiations of the functions at
:code:`ivy/functional/ivy/category_name.py`.

In order to explain how docstrings should be written, we will use :code:`ivy.tan` as an example.

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
    +    optional output array, for writing the result to. It must have a shape that the inputs
    +    broadcast to.

Because of this :code:`out` argument in the input, we also need to rename the :code:`out` argument in the return, which
is the default name used in the `Array API Standard`_. We change this to :code:`ret`:

.. code-block:: diff

    -out
    +ret

Next, we add a section in the docstring which explains that it has been modified from the version available in the
`Array API Standard`_:

.. code-block:: diff

    +This method conforms to the `Array API Standard
    +<https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    +`docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.tan.html>`_
    +in the standard.

Finally, **if** the function is *nestable*, then we add a simple explanation for this as follows:

.. code-block:: diff

    +Both the description and the type hints above assumes an array input for simplicity,
    +but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    +instances in place of any of the arguments.

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
        optional output array, for writing the result to. It must have a shape that the inputs
        broadcast to.

    Returns
    -------
    ret
        an array containing the tangent of each element in ``x``. The return must have a
        floating-point data type determined by :ref:`type-promotion`.

    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.tan.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

If the function that you are writing a docstring for is **not** in the `Array API Standard`_,
then you must simply follow this general template as closely as possible,
but instead you must use your own judgment when adding descriptions for the overall function,
and also for each of its arguments.