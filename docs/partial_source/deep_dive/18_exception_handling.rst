Exception Handling
==================

As Ivy is unifying various backends, various issues are seen during exception
handling:

#. each backend throws their own exceptions
#. exceptions thrown are backend-specific, therefore inconsistent

To unify the handling of exceptions and assertions, we created our custom
exception class and decorator, which will be explained further in the following
sections.


Ivy Exception Class
-------------------

Firstly, we created our main :code:`IvyException` class, inheriting from the
Python :code:`Exception` class.

(IvyException class code)

In cases where we require specific-purposed exception class, we will inherit
from the :code:`IvyException` class.
For example, the :code:`IvyBackendException` class is created to unify
backend exceptions.

(IvyBackendException class code)

@handle_exceptions Decorator
----------------------------

To ensure that all backend exceptions are caught properly, we add a decorator
to each function.

(decorator code)

In the code, all functions are wrapped in the :code:`try/except` block.
When a backend throws an exception, the block will catch it and raise
an :code:`IvyBackendException`.
This ensures that all exceptions are consistent.

(backend-specific errors vs. ivy exceptions from terminal)

From the example of the :code:`IvyBackendException` error message shown above,
the names of the backend and function of the original exception are stated
to provide clearer information.

Assertion Function
------------------

There are often conditions or limitations needed to ensure that a function
is working correctly.

Inconsistency is observed such as some functions:

#. use :code:`assert` for checks and throw :code:`AssertionError`, or
#. use :code:`if/elif/else` blocks and raise :code:`Exception`, :code:`ValueError`, etc.

To unify the behaviours, our policy is to use conditional blocks and
raise :code:`IvyException` whenever a check is required.
Moreover, to reduce code redundancy, we collect commonly-used conditions
into helper functions with custom parameters in :code:`ivy/assertions.py`.
This allows them to be reused and promotes cleaner code.

Let's look at an example!

(conditional blocks vs. helper function)
