.. _accuracy:

Accuracy
========

    Array API specification for minimum accuracy requirements.

Arithmetic Operations
---------------------

The results of element-wise arithmetic operations

-   ``+``
-   ``-``
-   ``*``
-   ``/``
-   ``%``

including the corresponding element-wise array APIs defined in this standard

-   add
-   subtract
-   multiply
-   divide

for floating-point operands must return the nearest representable value according to IEEE 754-2019 and a supported rounding mode. By default, the rounding mode should be ``roundTiesToEven`` (i.e., ties rounded toward the nearest value with an even least significant bit).

Mathematical Functions
----------------------

This specification does **not** precisely define the behavior of the following functions

-   acos
-   acosh
-   asin
-   asinh
-   atan
-   atan2
-   atanh
-   cos
-   cosh
-   exp
-   expm1
-   log
-   log1p
-   log2
-   log10
-   pow
-   sin
-   sinh
-   tan
-   tanh

except to require specific results for certain argument values that represent boundary cases of interest.

.. note::
   To help readers identify functions lacking precisely defined accuracy behavior, this specification uses the phrase "implementation-dependent approximation" in function descriptions.

For other argument values, these functions should compute approximations to the results of respective mathematical functions; however, this specification recognizes that array libraries may be constrained by underlying hardware and/or seek to optimize performance over absolute accuracy and, thus, allows some latitude in the choice of approximation algorithms.

Although the specification leaves the choice of algorithms to the implementation, this specification recommends (but does not specify) that implementations use the approximation algorithms for IEEE 754-2019 arithmetic contained in `FDLIBM <http://www.netlib.org/fdlibm>`_, the freely distributable mathematical library from Sun Microsystems, or some other comparable IEEE 754-2019 compliant mathematical library.

.. note::
   With exception of a few mathematical functions, returning results which are indistinguishable from correctly rounded infinitely precise results is difficult, if not impossible, to achieve due to the algorithms involved, the limits of finite-precision, and error propagation. However, this specification recognizes that numerical accuracy alignment among array libraries is desirable in order to ensure portability and reproducibility. Accordingly, for each mathematical function, the specification test suite includes test values which span a function's domain and reports the average and maximum deviation from either a designated standard implementation (e.g., an arbitrary precision arithmetic implementation) or an average computed across a subset of known array library implementations. Such reporting aids users who need to know how accuracy varies among libraries and developers who need to check the validity of their implementations.

Statistical Functions
---------------------

This specification does not specify accuracy requirements for statistical functions; however, this specification does expect that a conforming implementation of the array API standard will make a best-effort attempt to ensure that its implementations are theoretically sound and numerically robust.

.. note::
   In order for an array library to pass the specification test suite, an array library's statistical function implementations must satisfy certain bare-minimum accuracy requirements (e.g., accurate summation of a small set of positive integers). Unfortunately, imposing more rigorous accuracy requirements is not possible without severely curtailing possible implementation algorithms and unduly increasing implementation complexity.

Linear Algebra
--------------

This specification does not specify accuracy requirements for linear algebra functions; however, this specification does expect that a conforming implementation of the array API standard will make a best-effort attempt to ensure that its implementations are theoretically sound and numerically robust.