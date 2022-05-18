Adding Tests
============

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`test suite`: https://github.com/data-apis/array-api-tests
.. _`test_array_api`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/ivy_tests
.. _`for each backend`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/.github/workflows
.. _`hypothesis`: https://hypothesis.readthedocs.io/en/latest/

Array API Test Suite
--------------------

All functions which are present in the `Array API Standard`_ have a corresponding unit test in the
`test suite`_ for the standard.

The test suite is included in the ivy repository as a submodule in the folder `test_array_api`_,
which we keep updated with the upstream test suite.

The entire test suite is run independently `for each backend`_ on every push to the repo.
Therefore, every method which exists in the standard is thoroughly tested for adherence standard on a continuous basis.

Ivy Tests
---------

On top of the tests which we have taken directly from the test suite for the Array API Standard,
we also add our own tests.

This is for two reasons:

#. Many functions in Ivy are not present in the standard, and they need to be tested somewhere
#. The standard only mandates a subset of required behaviour. Almost all Ivy functions which are in the standard have additional required behaviour, which must also be tested

As is the case for the `test suite`_, we also make use of `hypothesis`_ for performing property based testing.

# ToDo: complete this section