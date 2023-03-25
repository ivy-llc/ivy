Sorting Functions
=================

  Array API specification for sorting functions.

A conforming implementation of the array API standard must provide and support the following functions adhering to the following conventions.

* Positional parameters must be `positional-only <https://www.python.org/dev/peps/pep-0570/>`_ parameters. Positional-only parameters have no externally-usable name. When a function accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order.
* Optional parameters must be `keyword-only <https://www.python.org/dev/peps/pep-3102/>`_ arguments.
* Unless stated otherwise, functions must support the data types defined in :ref:`data-types`.

.. note::

  For floating-point input arrays, the sort order of NaNs and signed zeros is unspecified and thus implementation-dependent.

  Implementations may choose to sort signed zeros (``-0 < +0``) or may choose to rely solely on value equality (``==``).

  Implementations may choose to sort NaNs (e.g., to the end or to the beginning of a returned array) or leave them in-place. Should an implementation sort NaNs, the sorting convention should be clearly documented in the conforming implementation's documentation.

  While defining a sort order for IEEE 754 floating-point numbers is recommended in order to facilitate reproducible and consistent sort results, doing so is not currently required by this specification.

.. currentmodule:: signatures.sorting_functions

Objects in API
--------------
..
  NOTE: please keep the functions in alphabetical order

.. autosummary::
   :toctree: generated
   :template: method.rst

   argsort
   sort
