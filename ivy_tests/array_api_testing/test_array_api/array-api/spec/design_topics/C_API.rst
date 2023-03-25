.. _C-API:

C API
=====

Use of a C API is out of scope for this array API, as mentioned in :ref:`Scope`.
There are a lot of libraries that do use such an API - in particular via Cython code
or via direct usage of the NumPy C API. When the maintainers of such libraries
want to use this array API standard to support multiple types of arrays, they
need a way to deal with that issue. This section aims to provide some guidance.

The assumption in the rest of this section is that performance matters for the library,
and hence the goal is to make other array types work without converting to a
``numpy.ndarray`` or another particular array type. If that's not the case (e.g. for a
visualization package), then other array types can simply be handled by converting
to the supported array type.

.. note::
   Often a zero-copy conversion to ``numpy.ndarray`` is possible, at least for CPU arrays.
   If that's the case, this may be a good way to support other array types.
   The main difficulty in that case will be getting the return array type right - however,
   this standard does provide a Python-level API for array construction that should allow
   doing this. A relevant question is if it's possible to know with
   certainty that a conversion will be zero-copy. This may indeed be
   possible, see :ref:`data-interchange`.


Example situations for C/Cython usage
-------------------------------------

Situation 1: a Python package that is mostly pure Python, with a limited number of Cython extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Projects in this situation include Statsmodels, scikit-bio and QuTiP

Main strategy: documentation. The functionality using Cython code will not support other array types (or only with conversion to/from ``numpy.ndarray``), which can be documented per function.


Situation 2: a Python package that contains a lot of Cython code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Projects in this situation include scikit-learn and scikit-image

Main strategy: add support for other array types *per submodule*. This keeps it manageable to explain to the user which functionality does and doesn't have support.

Longer term: specific support for particular array types (e.g. ``cupy.ndarray`` can be supported with Python-only code via ``cupy.ElementwiseKernel``).


Situation 3: a Python package that uses the NumPy or Python C API directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Projects in this situation include SciPy and Astropy

Strategy: similar to *situation 2*, but the number of submodules that can support all array types may be limited.


Device support
--------------

Supporting non-CPU array types in code using the C API or Cython seems problematic,
this almost inevitably will require custom device-specific code (e.g., CUDA, ROCm) or
something like JIT compilation with Numba.


Other longer-term approaches
----------------------------

Further Python API standardization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There may be cases where it makes sense to standardize additional sets of
functions, because they're important enough that array libraries tend to
reimplement them. An example of this may be *special functions*, as provided
by ``scipy.special``. Bessel and gamma functions for example are commonly
reimplemented by array libraries. This may avoid having to drop into a
particular implementation that does use a C API (e.g., one can then rely on
``arraylib.special.gamma`` rather than having to use ``scipy.special.gamma``).

HPy
~~~

`HPy <https://github.com/hpyproject/hpy>`_ is a new project that will provide a higher-level
C API and ABI than CPython offers. A Cython backend targeting HPy will be provided as well.

- Better PyPy support
- Universal ABI - single binary for all supported Python versions
- Cython backend generating HPy rather than CPython code

HPy isn't quite ready for mainstream usage today, but once it does it may
help make supporting multiple array libraries or adding non-CPU device
support to Cython more feasible.
