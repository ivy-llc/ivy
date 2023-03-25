.. _broadcasting:

Broadcasting
============

    Array API specification for broadcasting semantics.

Overview
--------

**Broadcasting** refers to the automatic (implicit) expansion of array dimensions to be of equal sizes without copying array data for the purpose of making arrays with different shapes have compatible shapes for element-wise operations.

Broadcasting facilitates user ergonomics by encouraging users to avoid unnecessary copying of array data and can **potentially** enable more memory-efficient element-wise operations through vectorization, reduced memory consumption, and cache locality.

Algorithm
---------

Given an element-wise operation involving two compatible arrays, an array having a singleton dimension (i.e., a dimension whose size is one) is broadcast (i.e., virtually repeated) across an array having a corresponding non-singleton dimension.

If two arrays are of unequal rank, the array having a lower rank is promoted to a higher rank by (virtually) prepending singleton dimensions until the number of dimensions matches that of the array having a higher rank.

The results of the element-wise operation must be stored in an array having a shape determined by the following algorithm.

#.  Let ``A`` and ``B`` both be arrays.

#.  Let ``shape1`` be a tuple describing the shape of array ``A``.

#.  Let ``shape2`` be a tuple describing the shape of array ``B``.

#.  Let ``N1`` be the number of dimensions of array ``A`` (i.e., the result of ``len(shape1)``).

#.  Let ``N2`` be the number of dimensions of array ``B`` (i.e., the result of ``len(shape2)``).

#.  Let ``N`` be the maximum value of ``N1`` and ``N2`` (i.e., the result of ``max(N1, N2)``).

#.  Let ``shape`` be a temporary list of length ``N`` for storing the shape of the result array.

#.  Let ``i`` be ``N-1``.

#.  Repeat, while ``i >= 0``

    #.  Let ``n1`` be ``N1 - N + i``.

    #.  If ``n1 >= 0``, let ``d1`` be the size of dimension ``n1`` for array ``A`` (i.e., the result of ``shape1[n1]``); else, let ``d1`` be ``1``.

    #.  Let ``n2`` be ``N2 - N + i``.

    #.  If ``n2 >= 0``, let ``d2`` be the size of dimension ``n2`` for array ``B`` (i.e., the result of ``shape2[n2]``); else, let ``d2`` be ``1``.

    #.  If ``d1 == 1``, then set the ``i``\th element of ``shape`` to ``d2``.

    #.  Else, if ``d2 == 1``, then

        -   set the ``i``\th element of ``shape`` to ``d1``.

    #.  Else, if ``d1 == d2``, then

        -   set the ``i``\th element of ``shape`` to ``d1``.

    #.  Else, throw an exception.

    #.  Set ``i`` to ``i-1``.

#.  Let ``tuple(shape)`` be the shape of the result array.

Examples
~~~~~~~~

The following examples demonstrate the application of the broadcasting algorithm for two compatible arrays.

::

   A      (4d array):  8 x 1 x 6 x 1
   B      (3d array):      7 x 1 x 5
  ---------------------------------
   Result (4d array):  8 x 7 x 6 x 5
   A      (2d array):  5 x 4
   B      (1d array):      1
   -------------------------
   Result (2d array):  5 x 4
   A      (2d array):  5 x 4
   B      (1d array):      4
   -------------------------
   Result (2d array):  5 x 4
   A      (3d array):  15 x 3 x 5
   B      (3d array):  15 x 1 x 5
   ------------------------------
   Result (3d array):  15 x 3 x 5
   A      (3d array):  15 x 3 x 5
   B      (2d array):       3 x 5
   ------------------------------
   Result (3d array):  15 x 3 x 5
   A      (3d array):  15 x 3 x 5
   B      (2d array):       3 x 1
   ------------------------------
   Result (3d array):  15 x 3 x 5


The following examples demonstrate array shapes which do **not** broadcast.

::

   A      (1d array):  3
   B      (1d array):  4           # dimension does not match

   A      (2d array):      2 x 1
   B      (3d array):  8 x 4 x 3   # second dimension does not match

   A      (3d array):  15 x 3 x 5
   B      (2d array):  15 x 3      # singleton dimensions can only be prepended, not appended

In-place Semantics
------------------

As implied by the broadcasting algorithm, in-place element-wise operations must not change the shape of the in-place array as a result of broadcasting.
