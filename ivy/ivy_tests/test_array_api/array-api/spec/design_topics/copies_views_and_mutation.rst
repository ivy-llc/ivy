.. _copyview-mutability:

Copy-view behaviour and mutability
==================================

Strided array implementations (e.g. NumPy, PyTorch, CuPy, MXNet) typically
have the concept of a "view", meaning an array containing data in memory that
belongs to another array (i.e. a different "view" on the original data).
Views are useful for performance reasons - not copying data to a new location
saves memory and is faster than copying - but can also affect the semantics
of code. This happens when views are combined with *mutating* operations.
This simple example illustrates that:

.. code-block:: python

   x = ones(1)
   y = x[:]  # `y` *may* be a view on the data of `x`
   y -= 1  # if `y` is a view, this modifies `x`

Code as simple as the above example will not be portable between array
libraries - for NumPy/PyTorch/CuPy/MXNet ``x`` will contain the value ``0``,
while for TensorFlow/JAX/Dask it will contain the value ``1``. The combination
of views and mutability is fundamentally problematic here if the goal is to
be able to write code with unambiguous semantics.

Views are necessary for getting good performance out of the current strided
array libraries. It is not always clear however when a library will return a
view, and when it will return a copy. This API standard does not attempt to
specify this - libraries can do either.

There are several types of operations that do in-place mutation of data
contained in arrays. These include:

1. Inplace operators (e.g. ``*=``)
2. Item assignment (e.g. ``x[0] = 1``)
3. Slice assignment (e.g., ``x[:2, :] = 3``)
4. The `out=` keyword present in some strided array libraries (e.g. ``sin(x, out=y)``)

Libraries like TensorFlow and JAX tend to support inplace operators, provide
alternative syntax for item and slice assignment (e.g. an ``update_index``
function or ``x.at[idx].set(y)``), and have no need for ``out=``.

A potential solution could be to make views read-only, or use copy-on-write
semantics. Both are hard to implement and would present significant issues
for backwards compatibility for current strided array libraries. Read-only
views would also not be a full solution, given that mutating the original
(base) array will also result in ambiguous semantics. Hence this API standard
does not attempt to go down this route.

Both inplace operators and item/slice assignment can be mapped onto
equivalent functional expressions (e.g. ``x[idx] = val`` maps to
``x.at[idx].set(val)``), and given that both inplace operators and item/slice
assignment are very widely used in both library and end user code, this
standard chooses to include them.

The situation with ``out=`` is slightly different - it's less heavily used, and
easier to avoid. It's also not an optimal API, because it mixes an
"efficiency of implementation" consideration ("you're allowed to do this
inplace") with the semantics of a function ("the output _must_ be placed into
this array). There are libraries that do some form of tracing or abstract
interpretation over a language that does not support mutation (to make
analysis easier); in those cases implementing ``out=`` with correct handling of
views may even be impossible to do. There's alternatives, for example the
donated arguments in JAX or working buffers in LAPACK, that allow the user to
express "you _may_ overwrite this data, do whatever is fastest". Given that
those alternatives aren't widely used in array libraries today, this API
standard chooses to (a) leave out ``out=``, and (b) not specify another method
of reusing arrays that are no longer needed as buffers.

This leaves the problem of the initial example - with this API standard it
remains possible to write code that will not work the same for all array
libraries. This is something that the user must be careful about.

.. note::
   It is recommended that users avoid any mutating operations when a view may be involved.
