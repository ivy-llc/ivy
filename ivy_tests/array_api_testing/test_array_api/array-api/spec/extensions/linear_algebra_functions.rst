.. _linear-algebra-extension:

Linear Algebra Extension
========================

    Array API specification for linear algebra functions.

A conforming implementation of the array API standard must provide and support the following functions adhering to the following conventions.

-   Positional parameters must be `positional-only <https://www.python.org/dev/peps/pep-0570/>`_ parameters. Positional-only parameters have no externally-usable name. When a function accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order.
-   Optional parameters must be `keyword-only <https://www.python.org/dev/peps/pep-3102/>`_ arguments.
-   Broadcasting semantics must follow the semantics defined in :ref:`broadcasting`.
-   Unless stated otherwise, functions must support the data types defined in :ref:`data-types`.
-   Unless stated otherwise, functions must adhere to the type promotion rules defined in :ref:`type-promotion`.
-   Unless stated otherwise, floating-point operations must adhere to IEEE 754-2019.

Design Principles
-----------------

A principal goal of this specification is to standardize commonly implemented interfaces among array libraries. While this specification endeavors to avoid straying too far from common practice, this specification does, with due restraint, seek to address design decisions arising more from historical accident than first principles. This is especially true for linear algebra APIs, which have arisen and evolved organically over time and have often been tied to particular underlying implementations (e.g., to BLAS and LAPACK).

Accordingly, the standardization process affords the opportunity to reduce interface complexity among linear algebra APIs by inferring and subsequently codifying common design themes, thus allowing more consistent APIs. What follows is the set of design principles governing the APIs which follow:

1.  **Batching**: if an operation is explicitly defined in terms of matrices (i.e., two-dimensional arrays), then the associated interface should support "batching" (i.e., the ability to perform the operation over a "stack" of matrices). Example operations include:

    -   ``inv``: computing the multiplicative inverse of a square matrix.
    -   ``cholesky``: performing Cholesky decomposition.
    -   ``matmul``: performing matrix multiplication.

2.  **Data types**: if an operation requires decimal operations and :ref:`type-promotion` semantics are undefined (e.g., as is the case for mixed-kind promotions), then the associated interface should be specified as being restricted to floating-point data types. While the specification uses the term "SHOULD" rather than "MUST", a conforming implementation of the array API standard should only ignore the restriction provided overly compelling reasons for doing so. Example operations which should be limited to floating-point data types include:

    -   ``inv``: computing the multiplicative inverse.
    -   ``slogdet``: computing the natural logarithm of the absolute value of the determinant.
    -   ``norm``: computing the matrix or vector norm.

    Certain operations are solely comprised of multiplications and additions. Accordingly, associated interfaces need not be restricted to floating-point data types. However, careful consideration should be given to overflow, and use of floating-point data types may be more prudent in practice. Example operations include:

    -   ``matmul``: performing matrix multiplication.
    -   ``trace``: computing the sum along the diagonal.
    -   ``cross``: computing the vector cross product.

    Lastly, certain operations may be performed independent of data type, and, thus, the associated interfaces should support all data types specified in this standard. Example operations include:

    -   ``matrix_transpose``: computing the transpose.
    -   ``diagonal``: returning the diagonal.

3.  **Return values**: if an interface has more than one return value, the interface should return a namedtuple consisting of each value.

    In general, interfaces should avoid polymorphic return values (e.g., returning an array **or** a namedtuple, dependent on, e.g., an optional keyword argument). Dedicated interfaces for each return value type are preferred, as dedicated interfaces are easier to reason about at both the implementation level and user level. Example interfaces which could be combined into a single overloaded interface, but are not, include:

    -   ``eig``: computing both eigenvalues and eignvectors.
    -   ``eigvals``: computing only eigenvalues.

4.  **Implementation agnosticism**: a standardized interface should eschew parameterization (including keyword arguments) biased toward particular implementations.

    Historically, at a time when all array computing happened on CPUs, BLAS and LAPACK underpinned most numerical computing libraries and environments. Naturally, language and library abstractions catered to the parameterization of those libraries, often exposing low-level implementation details verbatim in their higher-level interfaces, even if such choices would be considered poor or ill-advised by today's standards (e.g., NumPy's use of `UPLO` in `eigh`). However, the present day is considerably different. While still important, BLAS and LAPACK no longer hold a monopoly over linear algebra operations, especially given the proliferation of devices and hardware on which such operations must be performed. Accordingly, interfaces must be conservative in the parameterization they support in order to best ensure universality. Such conservatism applies even to performance optimization parameters afforded by certain hardware.

5.  **Orthogonality**: an interface should have clearly defined and delineated functionality which, ideally, has no overlap with the functionality of other interfaces in the specification. Providing multiple interfaces which can all perform the same operation creates unnecessary confusion regarding interface applicability (i.e., which interface is best at which time) and decreases readability of both library and user code. Where overlap is possible, the specification must be parsimonious in the number of interfaces, ensuring that each interface provides a unique and compelling abstraction. Examples of related interfaces which provide distinct levels of abstraction (and generality) include:

    -   ``vecdot``: computing the dot product of two vectors.
    -   ``matmul``: performing matrix multiplication (including between two vectors and thus the dot product).
    -   ``tensordot``: computing tensor contractions (generalized sum-products).
    -   ``einsum``: expressing operations in terms of Einstein summation convention, including dot products and tensor contractions.

    The above can be contrasted with, e.g., NumPy, which provides the following interfaces for computing the dot product or related operations:

    -   ``dot``: dot product, matrix multiplication, and tensor contraction.
    -   ``inner``: dot product.
    -   ``vdot``: dot product with flattening and complex conjugation.
    -   ``multi_dot``: chained dot product.
    -   ``tensordot``: tensor contraction.
    -   ``matmul``: matrix multiplication (dot product for two vectors).
    -   ``einsum``: Einstein summation convention.

    where ``dot`` is overloaded based on input array dimensionality and ``vdot`` and ``inner`` exhibit a high degree of overlap with other interfaces. By consolidating interfaces and more clearly delineating behavior, this specification aims to ensure that each interface has a unique purpose and defined use case.

.. currentmodule:: signatures.linalg

Objects in API
--------------
..
  NOTE: please keep the functions in alphabetical order

.. autosummary::
   :toctree: generated
   :template: method.rst

   cholesky
   cross
   det
   diagonal
   eigh
   eigvalsh
   inv
   matmul
   matrix_norm
   matrix_power
   matrix_rank
   matrix_transpose
   outer
   pinv
   qr
   slogdet
   solve
   svd
   svdvals
   tensordot
   trace
   vecdot
   vector_norm
