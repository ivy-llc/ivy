.. _array-object:

Array object
============

    Array API specification for array object attributes and methods.

A conforming implementation of the array API standard must provide and support an array object having the following attributes and methods adhering to the following conventions.

*   Positional parameters must be `positional-only <https://www.python.org/dev/peps/pep-0570/>`_ parameters. Positional-only parameters have no externally-usable name. When a method accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order.
*   Optional parameters must be `keyword-only <https://www.python.org/dev/peps/pep-3102/>`_ arguments.
*   Broadcasting semantics must follow the semantics defined in :ref:`broadcasting`.
*   Unless stated otherwise, methods must support the data types defined in :ref:`data-types`.
*   Unless stated otherwise, methods must adhere to the type promotion rules defined in :ref:`type-promotion`.
*   Unless stated otherwise, floating-point operations must adhere to IEEE 754-2019.

Furthermore, a conforming implementation of the array API standard must support array objects of arbitrary rank ``N`` (i.e., number of dimensions), where ``N`` is greater than or equal to zero.

.. note::
    Conforming implementations must support zero-dimensional arrays.

    Apart from array object attributes, such as ``ndim``, ``device``, and ``dtype``, all operations in this standard return arrays (or tuples of arrays), including those operations, such as ``mean``, ``var``, and ``std``, from which some common array libraries (e.g., NumPy) return scalar values.

    *Rationale: always returning arrays is necessary to (1) support accelerator libraries where non-array return values could force device synchronization and (2) support delayed execution models where an array represents a future value.*

-------------------------------------------------

.. _operators:

Operators
---------

A conforming implementation of the array API standard must provide and support an array object supporting the following Python operators.

Arithmetic Operators
~~~~~~~~~~~~~~~~~~~~

A conforming implementation of the array API standard must provide and support an array object supporting the following Python arithmetic operators.

-   ``+x``: :meth:`.array.__pos__`

    -   `operator.pos(x) <https://docs.python.org/3/library/operator.html#operator.pos>`_
    -   `operator.__pos__(x) <https://docs.python.org/3/library/operator.html#operator.__pos__>`_

-   `-x`: :meth:`.array.__neg__`

    -   `operator.neg(x) <https://docs.python.org/3/library/operator.html#operator.neg>`_
    -   `operator.__neg__(x) <https://docs.python.org/3/library/operator.html#operator.__neg__>`_

-   `x1 + x2`: :meth:`.array.__add__`

    -   `operator.add(x1, x2) <https://docs.python.org/3/library/operator.html#operator.add>`_
    -   `operator.__add__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__add__>`_

-   `x1 - x2`: :meth:`.array.__sub__`

    -   `operator.sub(x1, x2) <https://docs.python.org/3/library/operator.html#operator.sub>`_
    -   `operator.__sub__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__sub__>`_

-   `x1 * x2`: :meth:`.array.__mul__`

    -   `operator.mul(x1, x2) <https://docs.python.org/3/library/operator.html#operator.mul>`_
    -   `operator.__mul__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__mul__>`_

-   `x1 / x2`: :meth:`.array.__truediv__`

    -   `operator.truediv(x1,x2) <https://docs.python.org/3/library/operator.html#operator.truediv>`_
    -   `operator.__truediv__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__truediv__>`_

-   `x1 // x2`: :meth:`.array.__floordiv__`

    -   `operator.floordiv(x1, x2) <https://docs.python.org/3/library/operator.html#operator.floordiv>`_
    -   `operator.__floordiv__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__floordiv__>`_

-   `x1 % x2`: :meth:`.array.__mod__`

    -   `operator.mod(x1, x2) <https://docs.python.org/3/library/operator.html#operator.mod>`_
    -   `operator.__mod__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__mod__>`_

-   `x1 ** x2`: :meth:`.array.__pow__`

    -   `operator.pow(x1, x2) <https://docs.python.org/3/library/operator.html#operator.pow>`_
    -   `operator.__pow__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__pow__>`_

Arithmetic operators should be defined for arrays having numeric data types.

Array Operators
~~~~~~~~~~~~~~~

A conforming implementation of the array API standard must provide and support an array object supporting the following Python array operators.

-   `x1 @ x2`: :meth:`.array.__matmul__`

    -   `operator.matmul(x1, x2) <https://docs.python.org/3/library/operator.html#operator.matmul>`_
    -   `operator.__matmul__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__matmul__>`_

The matmul ``@`` operator should be defined for arrays having numeric data types.

Bitwise Operators
~~~~~~~~~~~~~~~~~

A conforming implementation of the array API standard must provide and support an array object supporting the following Python bitwise operators.

-   `~x`: :meth:`.array.__invert__`

    -   `operator.inv(x) <https://docs.python.org/3/library/operator.html#operator.inv>`_
    -   `operator.invert(x) <https://docs.python.org/3/library/operator.html#operator.invert>`_
    -   `operator.__inv__(x) <https://docs.python.org/3/library/operator.html#operator.__inv__>`_
    -   `operator.__invert__(x) <https://docs.python.org/3/library/operator.html#operator.__invert__>`_

-   `x1 & x2`: :meth:`.array.__and__`

    -   `operator.and(x1, x2) <https://docs.python.org/3/library/operator.html#operator.and>`_
    -   `operator.__and__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__and__>`_

-   `x1 | x2`: :meth:`.array.__or__`

    -   `operator.or(x1, x2) <https://docs.python.org/3/library/operator.html#operator.or>`_
    -   `operator.__or__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__or__>`_

-   `x1 ^ x2`: :meth:`.array.__xor__`

    -   `operator.xor(x1, x2) <https://docs.python.org/3/library/operator.html#operator.xor>`_
    -   `operator.__xor__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__xor__>`_

-   `x1 << x2`: :meth:`.array.__lshift__`

    -   `operator.lshift(x1, x2) <https://docs.python.org/3/library/operator.html#operator.lshift>`_
    -   `operator.__lshift__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__lshift__>`_

-   `x1 >> x2`: :meth:`.array.__rshift__`

    -   `operator.rshift(x1, x2) <https://docs.python.org/3/library/operator.html#operator.rshift>`_
    -   `operator.__rshift__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__rshift__>`_

Bitwise operators should be defined for arrays having integer and boolean data types.

Comparison Operators
~~~~~~~~~~~~~~~~~~~~

A conforming implementation of the array API standard must provide and support an array object supporting the following Python comparison operators.

-   `x1 < x2`: :meth:`.array.__lt__`

    -   `operator.lt(x1, x2) <https://docs.python.org/3/library/operator.html#operator.lt>`_
    -   `operator.__lt__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__lt__>`_

-   `x1 <= x2`: :meth:`.array.__le__`

    -   `operator.le(x1, x2) <https://docs.python.org/3/library/operator.html#operator.le>`_
    -   `operator.__le__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__le__>`_

-   `x1 > x2`: :meth:`.array.__gt__`

    -   `operator.gt(x1, x2) <https://docs.python.org/3/library/operator.html#operator.gt>`_
    -   `operator.__gt__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__gt__>`_

-   `x1 >= x2`: :meth:`.array.__ge__`

    -   `operator.ge(x1, x2) <https://docs.python.org/3/library/operator.html#operator.ge>`_
    -   `operator.__ge__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__ge__>`_

-   `x1 == x2`: :meth:`.array.__eq__`

    -   `operator.eq(x1, x2) <https://docs.python.org/3/library/operator.html#operator.eq>`_
    -   `operator.__eq__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__eq__>`_

-   `x1 != x2`: :meth:`.array.__ne__`

    -   `operator.ne(x1, x2) <https://docs.python.org/3/library/operator.html#operator.ne>`_
    -   `operator.__ne__(x1, x2) <https://docs.python.org/3/library/operator.html#operator.__ne__>`_

Comparison operators should be defined for arrays having any data type.

In-place Operators
~~~~~~~~~~~~~~~~~~

A conforming implementation of the array API standard must provide and support an array object supporting the following in-place Python operators.

An in-place operation must not change the data type or shape of the in-place array as a result of :ref:`type-promotion` or :ref:`broadcasting`.

An in-place operation must have the same behavior (including special cases) as its respective binary (i.e., two operand, non-assignment) operation. For example, after in-place addition ``x1 += x2``, the modified array ``x1`` must always equal the result of the equivalent binary arithmetic operation ``x1 = x1 + x2``.

.. note::
    In-place operators must be supported as discussed in :ref:`copyview-mutability`.

Arithmetic Operators
""""""""""""""""""""

-   ``+=``. May be implemented via ``__iadd__``.
-   ``-=``. May be implemented via ``__isub__``.
-   ``*=``. May be implemented via ``__imul__``.
-   ``/=``. May be implemented via ``__itruediv__``.
-   ``//=``. May be implemented via ``__ifloordiv__``.
-   ``**=``. May be implemented via ``__ipow__``.
-   ``%=``. May be implemented via ``__imod__``.

Array Operators
"""""""""""""""

-   ``@=``. May be implemented via ``__imatmul__``.

Bitwise Operators
"""""""""""""""""

-   ``&=``. May be implemented via ``__iand__``.
-   ``|=``. May be implemented via ``__ior__``.
-   ``^=``. May be implemented via ``__ixor__``.
-   ``<<=``. May be implemented via ``__ilshift__``.
-   ``>>=``. May be implemented via ``__irshift__``.

Reflected Operators
~~~~~~~~~~~~~~~~~~~

A conforming implementation of the array API standard must provide and support an array object supporting the following reflected operators.

The results of applying reflected operators must match their non-reflected equivalents.

.. note::
    All operators for which ``array <op> scalar`` is implemented must have an equivalent reflected operator implementation.

Arithmetic Operators
""""""""""""""""""""

-   ``__radd__``
-   ``__rsub__``
-   ``__rmul__``
-   ``__rtruediv__``
-   ``__rfloordiv__``
-   ``__rpow__``
-   ``__rmod__``

Array Operators
"""""""""""""""

-   ``__rmatmul__``

Bitwise Operators
"""""""""""""""""

-   ``__rand__``
-   ``__ror__``
-   ``__rxor__``
-   ``__rlshift__``
-   ``__rrshift__``

-------------------------------------------------

.. currentmodule:: signatures.array_object

Attributes
----------
..
  NOTE: please keep the attributes in alphabetical order


.. autosummary::
   :toctree: generated
   :template: property.rst

   array.dtype
   array.device
   array.mT
   array.ndim
   array.shape
   array.size
   array.T

-------------------------------------------------

Methods
-------
..
  NOTE: please keep the methods in alphabetical order


.. autosummary::
   :toctree: generated
   :template: property.rst

   array.__abs__
   array.__add__
   array.__and__
   array.__array_namespace__
   array.__bool__
   array.__dlpack__
   array.__dlpack_device__
   array.__eq__
   array.__float__
   array.__floordiv__
   array.__ge__
   array.__getitem__
   array.__gt__
   array.__index__
   array.__int__
   array.__invert__
   array.__le__
   array.__lshift__
   array.__lt__
   array.__matmul__
   array.__mod__
   array.__mul__
   array.__ne__
   array.__neg__
   array.__or__
   array.__pos__
   array.__pow__
   array.__rshift__
   array.__setitem__
   array.__sub__
   array.__truediv__
   array.__xor__
   array.to_device
