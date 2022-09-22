.. _element-wise-functions:

Element-wise Functions
======================

    Array API specification for element-wise functions.

A conforming implementation of the array API standard must provide and support the following functions adhering to the following conventions.

-   Positional parameters must be `positional-only <https://www.python.org/dev/peps/pep-0570/>`_ parameters. Positional-only parameters have no externally-usable name. When a function accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order.
-   Optional parameters must be `keyword-only <https://www.python.org/dev/peps/pep-3102/>`_ arguments.
-   Broadcasting semantics must follow the semantics defined in :ref:`broadcasting`.
-   Unless stated otherwise, functions must support the data types defined in :ref:`data-types`.
-   Functions may only be required for a subset of input data type. Libraries may choose to implement functions for additional data types, but that behavior is not required by the specification. See :ref:`data-type-categories`.
-   Unless stated otherwise, functions must adhere to the type promotion rules defined in :ref:`type-promotion`.
-   Unless stated otherwise, floating-point operations must adhere to IEEE 754-2019.
-   Unless stated otherwise, element-wise mathematical functions must satisfy the minimum accuracy requirements defined in :ref:`accuracy`.

Objects in API
--------------

.. currentmodule:: signatures.elementwise_functions

..
  NOTE: please keep the functions in alphabetical order

.. autosummary::
   :toctree: generated
   :template: method.rst

   abs
   acos
   acosh
   add
   asin
   asinh
   atan
   atan2
   atanh
   bitwise_and
   bitwise_left_shift
   bitwise_invert
   bitwise_or
   bitwise_right_shift
   bitwise_xor
   ceil
   cos
   cosh
   divide
   equal
   exp
   expm1
   floor
   floor_divide
   greater
   greater_equal
   isfinite
   isinf
   isnan
   less
   less_equal
   log
   log1p
   log2
   log10
   logaddexp
   logical_and
   logical_not
   logical_or
   logical_xor
   multiply
   negative
   not_equal
   positive
   pow
   remainder
   round
   sign
   sin
   sinh
   square
   sqrt
   subtract
   tan
   tanh
   trunc
