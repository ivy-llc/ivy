.. _type-promotion:

Type Promotion Rules
====================

    Array API specification for type promotion rules.

Type promotion rules can be understood at a high level from the following diagram:

.. image:: /_static/images/dtype_promotion_lattice.png
    :target: Type promotion diagram

*Type promotion diagram. Promotion between any two types is given by their join on this lattice. Only the types of participating arrays matter, not their values. Dashed lines indicate that behavior for Python scalars is undefined on overflow. Boolean, integer and floating-point dtypes are not connected, indicating mixed-kind promotion is undefined.*

Rules
-----

A conforming implementation of the array API standard must implement the following type promotion rules governing the common result type for two **array** operands during an arithmetic operation.

A conforming implementation of the array API standard may support additional type promotion rules beyond those described in this specification.

.. note::
   Type codes are used here to keep tables readable; they are not part of the standard. In code, use the data type objects specified in :ref:`data-types` (e.g., ``int16`` rather than ``'i2'``).

..
  Note: please keep table columns aligned

The following type promotion tables specify the casting behavior for operations involving two array operands. When more than two array operands participate, application of the promotion tables is associative (i.e., the result does not depend on operand order).

Signed integer type promotion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------+----+----+----+----+
|        | i1 | i2 | i4 | i8 |
+========+====+====+====+====+
| **i1** | i1 | i2 | i4 | i8 |
+--------+----+----+----+----+
| **i2** | i2 | i2 | i4 | i8 |
+--------+----+----+----+----+
| **i4** | i4 | i4 | i4 | i8 |
+--------+----+----+----+----+
| **i8** | i8 | i8 | i8 | i8 |
+--------+----+----+----+----+

where

-   **i1**: 8-bit signed integer (i.e., ``int8``)
-   **i2**: 16-bit signed integer (i.e., ``int16``)
-   **i4**: 32-bit signed integer (i.e., ``int32``)
-   **i8**: 64-bit signed integer (i.e., ``int64``)

Unsigned integer type promotion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------+----+----+----+----+
|        | u1 | u2 | u4 | u8 |
+========+====+====+====+====+
| **u1** | u1 | u2 | u4 | u8 |
+--------+----+----+----+----+
| **u2** | u2 | u2 | u4 | u8 |
+--------+----+----+----+----+
| **u4** | u4 | u4 | u4 | u8 |
+--------+----+----+----+----+
| **u8** | u8 | u8 | u8 | u8 |
+--------+----+----+----+----+

where

-   **u1**: 8-bit unsigned integer (i.e., ``uint8``)
-   **u2**: 16-bit unsigned integer (i.e., ``uint16``)
-   **u4**: 32-bit unsigned integer (i.e., ``uint32``)
-   **u8**: 64-bit unsigned integer (i.e., ``uint64``)

Mixed unsigned and signed integer type promotion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------+----+----+----+
|        | u1 | u2 | u4 |
+========+====+====+====+
| **i1** | i2 | i4 | i8 |
+--------+----+----+----+
| **i2** | i2 | i4 | i8 |
+--------+----+----+----+
| **i4** | i4 | i4 | i8 |
+--------+----+----+----+
| **i8** | i8 | i8 | i8 |
+--------+----+----+----+

Floating-point type promotion table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------+----+----+
|        | f4 | f8 |
+========+====+====+
| **f4** | f4 | f8 |
+--------+----+----+
| **f8** | f8 | f8 |
+--------+----+----+

where

-   **f4**: single-precision (32-bit) floating-point number (i.e., ``float32``)
-   **f8**: double-precision (64-bit) floating-point number (i.e., ``float64``)

Notes
~~~~~

-   Type promotion rules must apply when determining the common result type for two **array** operands during an arithmetic operation, regardless of array dimension. Accordingly, zero-dimensional arrays must be subject to the same type promotion rules as dimensional arrays.
-   Type promotion of non-numerical data types to numerical data types is unspecified (e.g., ``bool`` to ``intxx`` or ``floatxx``).

.. note::
   Mixed integer and floating-point type promotion rules are not specified because behavior varies between implementations.

Mixing arrays with Python scalars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Python scalars (i.e., instances of ``bool``, ``int``, ``float``) together with arrays must be supported for:

-   ``array <op> scalar``
-   ``scalar <op> array``

where ``<op>`` is a built-in operator (including in-place operators, but excluding the matmul ``@`` operator; see :ref:`operators` for operators supported by the array object) and ``scalar`` has a type and value compatible with the array data type:

-   a Python ``bool`` for a ``bool`` array data type.
-   a Python ``int`` within the bounds of the given data type for integer array :ref:`data-types`.
-   a Python ``int`` or ``float`` for floating-point array data types.

Provided the above requirements are met, the expected behavior is equivalent to:

1.  Convert the scalar to zero-dimensional array with the same data type as that of the array used in the expression.
2.  Execute the operation for ``array <op> 0-D array`` (or ``0-D array <op> array`` if ``scalar`` was the left-hand argument).

.. note::
   Behavior is not specified when mixing a Python ``float`` and an array with an integer data type; this may give ``float32``, ``float64``, or raise an exception. Behavior is implementation-specific.

   The behavior is also not specified for integers outside of the bounds of a given integer data type. Integers outside of bounds may result in overflow or an error.
