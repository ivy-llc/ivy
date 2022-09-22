Manipulation Functions
======================

    Array API specification for manipulating arrays.

A conforming implementation of the array API standard must provide and support the following functions adhering to the following conventions.

-   Positional parameters must be `positional-only <https://www.python.org/dev/peps/pep-0570/>`_ parameters. Positional-only parameters have no externally-usable name. When a function accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order.
-   Optional parameters must be `keyword-only <https://www.python.org/dev/peps/pep-3102/>`_ arguments.
-   Unless stated otherwise, functions must adhere to the type promotion rules defined in :ref:`type-promotion`.

Objects in API
--------------

.. currentmodule:: signatures.manipulation_functions

..
  NOTE: please keep the functions in alphabetical order

.. autosummary::
   :toctree: generated
   :template: method.rst

   concat
   expand_dims
   flip
   permute_dims
   reshape
   roll
   squeeze
   stack
