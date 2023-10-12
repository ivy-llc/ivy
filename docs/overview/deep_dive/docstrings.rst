Docstrings
==========

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`spec/API_specification/array_api`: https://github.com/data-apis/array-api/blob/main
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`docstrings channel`: https://discord.com/channels/799879767196958751/982738313897197600
.. _`docstrings forum`: https://discord.com/channels/799879767196958751/1028297612408913982


All functions in the Ivy API at :mod:`ivy/functional/ivy/category_name.py` should have full and thorough docstrings.
In contrast, all backend implementations at :mod:`ivy/functional/backends/backend_name/category_name.py` should not have any docstrings, on account that these are effectively just different instantiations of the functions at :mod:`ivy/functional/ivy/category_name.py`.

In order to explain how docstrings should be written, we will use :func:`ivy.tan` as an example.

Firstly, if the function exists in the `Array API Standard`_, then we start with the corresponding docstring as a template.
These docstrings can be found under `spec/API_specification/array_api`_.

Important: you should open the file in **raw** format.
If you copy directly from the file preview on GitHub before clicking **raw**, then the newlines will **not** be copied over, and the docstring will rendering incorrectly in the online docs.

The `Array API Standard`_ docstring for :code:`tan` is as follows:

.. parsed-literal::

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

This is a good starting point.
But we need to make some changes.
Firstly, given that we are using type hints, repeating all of the types also in the docs would be a needless duplication.
Therefore, we simply remove all type info from the docstring like so:

.. code-block:: diff

    -x: array
    +x
    -out: array
    +out

The `Array API Standard`_ defines a subset of behaviour that each function must adhere to.
Ivy extends many of these functions with additional behaviour and arguments.
In the case of :func:`ivy.tan`, there is also the argument :code:`out` which needs to be added to the docstring, like so:

.. code-block:: diff

    +out
    +    optional output array, for writing the result to. It must have a shape that the inputs
    +    broadcast to.

Because of this :code:`out` argument in the input, we also need to rename the :code:`out` argument in the return, which is the default name used in the `Array API Standard`_.
We change this to :code:`ret`:

.. code-block:: diff

    -out
    +ret

Next, we add a section in the docstring which explains that it has been modified from the version available in the
`Array API Standard`_:

.. code-block:: diff

    +This function conforms to the `Array API Standard
    +<https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    +`docstring <https://data-apis.org/array-api/latest/API_specification/generated/array_api.tan.html>`_
    +in the standard.

Finally, **if** the function is *nestable*, then we add a simple explanation for this as follows:

.. code-block:: diff

    +Both the description and the type hints above assumes an array input for simplicity,
    +but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    +instances in place of any of the arguments.

Following these changes, the new docstring is as follows:

.. parsed-literal::

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

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/array_api.tan.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

If the function that you are writing a docstring for is **not** in the `Array API Standard`_, then you must simply follow this general template as closely as possible, but instead you must use your own judgment when adding descriptions for the overall function, and also for each of its arguments.

**Classes**

The instance methods in :class:`ivy.Array` and :class:`ivy.Container` which directly wrap a function in the functional API do not require thorough docstrings, on account that these instance methods require no explanation beyond that provided in the docstring for the wrapped function.

Therefore, these docstrings should all simply contain the following text:

.. code-block:: python

    ivy.<Array|Container> <instance|special|reverse special> method variant of ivy.<func_name>. This method simply wraps the
    function, and so the docstring for ivy.<func_name> also applies to this method
    with minimal changes.

    Parameters
    ----------
    <parameters with their description>

    Returns
    -------
    <return value with its description>

The exception to this is :class:`ivy.Container` :code:`special` method docstrings,
which should instead use the following text, as these do not *directly* wrap a function
in Ivy's functional API, but rather wrap the pure operator functions themselves,
which can be called on any types that support the corresponding special methods:

.. parsed-literal::

    ivy.Container <special|reverse special> method for the <operator_name> operator,
    calling :code:`operator.<operator_name>` for each of the corresponding leaves of
    the two containers.

    Parameters
    ----------
    <parameters with their description>

    Returns
    -------
    <return value with its description>

Let's take :func:`ivy.add` as an example.
The docstring for `ivy.add <https://github.com/unifyai/ivy/blob/04766790a518ecde380cb6eeb05aa89cf5acdbfd/ivy/functional/ivy/elementwise.py#L191>`_ is thorough, as explained above.
However, the docstrings for `ivy.Array.add <https://github.com/unifyai/ivy/blob/04766790a518ecde380cb6eeb05aa89cf5acdbfd/ivy/array/elementwise.py#L36>`_, `ivy.Container.add <https://github.com/unifyai/ivy/blob/04766790a518ecde380cb6eeb05aa89cf5acdbfd/ivy/container/elementwise.py#L209>`_ all follow the succinct pattern outlined above.
Likewise, the docstrings for the special methods `ivy.Array.__add__ <https://github.com/unifyai/ivy/blob/04766790a518ecde380cb6eeb05aa89cf5acdbfd/ivy/array/array.py#L310>`_, `ivy.Array.__radd__ <https://github.com/unifyai/ivy/blob/04766790a518ecde380cb6eeb05aa89cf5acdbfd/ivy/array/array.py#L359>`_, `ivy.Container.__add__ <https://github.com/unifyai/ivy/blob/04766790a518ecde380cb6eeb05aa89cf5acdbfd/ivy/container/container.py#L106>`_, and `ivy.Container.__radd__ <https://github.com/unifyai/ivy/blob/04766790a518ecde380cb6eeb05aa89cf5acdbfd/ivy/container/container.py#L171>`_, also follow the succinct pattern outlined above.
Note that these docstrings all *also* include examples, which we will cover in the next section.

For all other classes, such as the various layers at :code:`ivy/ivy/stateful/layers`, then we should add full and thorough docstrings for both the **contstructor** and also **all methods**.

This is the case even when the class directly wraps a function in the functional API.
For example, the class `ivy.Linear <https://github.com/unifyai/ivy/blob/51c23694c2f51e88caef0f382f200b195f8458b5/ivy/stateful/layers.py#L13>`_ wraps the function `ivy.linear <https://github.com/unifyai/ivy/blob/51c23694c2f51e88caef0f382f200b195f8458b5/ivy/functional/ivy/layers.py#L22>`_, but does so in a stateful manner with the variables stored internally in the instance of the class.
Even though the :class:`ivy.Linear` class wraps :func:`ivy.linear` in the forward pass defined in `ivy.Linear._forward <https://github.com/unifyai/ivy/blob/51c23694c2f51e88caef0f382f200b195f8458b5/ivy/stateful/layers.py#L84>`_, the function signatures of :func:`ivy.linear` and :meth:`ivy.Linear._forward` are still quite distinct, with the former including all trainable variables explicitly, and the latter having these implicit as internal instance attributes of the class.

Therefore, with the exception of the :class:`ivy.Array` and :class:`ivy.Container` methods which directly wrap functions in the functional API, we should always add full and thorough docstrings to all methods of all other classes in Ivy, including cases where these also directly wrap functions in the functional API.

**Round Up**

These examples should hopefully give you a good understanding of what is required when adding docstings.

If you have any questions,please feel free to reach out on `discord`_ in the `docstrings channel`_ or in the `docstrings forum`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/TnshJ8swuJM" class="video">
    </iframe>
