.. _function-and-method-signatures:

Function and method signatures
==============================

Function signatures in this standard adhere to the following:

1. Positional parameters must be `positional-only <https://www.python.org/dev/peps/pep-0570/>`_ parameters.
   Positional-only parameters have no externally-usable name. When a function
   accepting positional-only parameters is called, positional arguments are
   mapped to these parameters based solely on their order.

   *Rationale: existing libraries have incompatible conventions, and using names
   of positional parameters is not normal/recommended practice.*

   .. note::

    Positional-only parameters are only available in Python >= 3.8. Libraries
    still supporting 3.7 or 3.6 may consider making the API standard-compliant
    namespace >= 3.8. Alternatively, they can add guidance to their users in the
    documentation to use the functions as if they were positional-only.

2. Optional parameters must be `keyword-only <https://www.python.org/dev/peps/pep-3102/>`_ arguments.

   *Rationale: this leads to more readable code, and it makes it easier to
   evolve an API over time by adding keywords without having to worry about
   keyword order.*

3. For functions that have a single positional array parameter, that parameter
   is called ``x``. For functions that have multiple array parameters, those
   parameters are called ``xi`` with ``i = 1, 2, ...`` (i.e., ``x1``, ``x2``).

4. Type annotations are left out of the signatures themselves for readability; however,
   they are added to individual parameter descriptions. For code which aims to
   adhere to the standard, adding type annotations is strongly recommended.

A function signature and description will look like:

::

   funcname(x1, x2, /, *, key1=-1, key2=None) -> out:
       Parameters

        x1 : array
            description
        x2 : array
            description
        key1 : int
            description
        key2 : Optional[str]
            description

        Returns

        out : array
            description


Method signatures will follow the same conventions modulo the addition of ``self``.
