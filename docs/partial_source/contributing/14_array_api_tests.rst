Array API Tests
===============

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`test suite`: https://github.com/data-apis/array-api-tests
.. _`test_array_api`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/ivy_tests
.. _`for each backend`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/.github/workflows

All functions which are present in the `Array API Standard`_ have a corresponding unit test in the
`test suite`_ for the standard.

The test suite is included in the ivy repository as a submodule in the folder `test_array_api`_,
which we keep updated with the upstream test suite.

The entire test suite is run independently `for each backend`_ on every push to the repo.
Therefore, every method which exists in the standard is thoroughly tested for
adherence to the standard on a continuous basis.

You will need to make sure the Array API tests are passing for each backend framework if/when making any changes to Ivy
functions which are part of the standard.