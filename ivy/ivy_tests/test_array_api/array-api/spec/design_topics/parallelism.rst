Parallelism
===========

Parallelism is mostly, but not completely, an execution or runtime concern
rather than an API concern. Execution semantics are out of scope for this API
standard, and hence won't be discussed further here. The API related part
involves how libraries allow users to exercise control over the parallelism
they offer, such as:

- Via environment variables. This is the method of choice for BLAS libraries and libraries using OpenMP.
- Via a keyword to individual functions or methods. Examples include the ``n_jobs`` keyword used in scikit-learn and the ``workers`` keyword used in SciPy.
- Build-time settings to enable a parallel or distributed backend.
- Via letting the user set chunk sizes. Dask uses this approach.

When combining multiple libraries, one has to deal with auto-parallelization
semantics and nested parallelism. Two things that could help improve the
coordination of parallelization behavior in a stack of Python libraries are:

1. A common API pattern for enabling parallelism
2. A common library providing a parallelization layer

Option (1) may possibly fit in a future version of this array API standard.
`array-api issue 4 <https://github.com/data-apis/array-api/issues/4>`_ contains
more detailed discussion on the topic of parallelism.