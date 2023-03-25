Static typing
=============

Good support for static typing both in array libraries and array-consuming
code is desirable. Therefore the exact type or set of types for each
parameter, keyword and return value is specified for functions and methods -
see :ref:`function-and-method-signatures`. That section specifies arrays
simply as ``array``; what that means is dealt with in this section.

Introducing type annotations in libraries became more relevant only when
Python 2.7 support was dropped at the start of 2020. As a consequence, using
type annotations with array libraries is largely still a work in progress.
This version of the API standard does not deal with trying to type *array
properties* like shape, dimensionality or dtype, because that's not a solved
problem in individual array libraries yet.

An ``array`` type annotation can mean either the type of one specific array
object, or some superclass or typing Protocol - as long as it is consistent
with the array object specified in :ref:`array-object`. To illustrate by
example:

.. code-block:: python

   # `Array` is a particular class in the library
   def sin(x: Array, / ...) -> Array:
       ...

and

.. code-block:: python

   # There's some base class `_BaseArray`, and there may be multiple
   # array subclasses inside the library
   A = TypeVar('A', bound=_BaseArray)
   def sin(x: A, / ...) -> A:
       ...

should both be fine. There may be other variations possible. Also note that
this standard does not require that input and output array types are the same
(they're expected to be defined in the same library though). Given that
array libraries don't have to be aware of other types of arrays defined in
other libraries (see :ref:`assumptions-dependencies`), this should be enough
for a single array library.

That said, an array-consuming library aiming to support multiple array types
may need more - for example a protocol to enable structural subtyping. This
API standard currently takes the position that it does not provide any
reference implementation or package that can or should be relied on at
runtime, hence no such protocol is defined here. This may be dealt with in a
future version of this standard.
