Array API Tests
===============

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`test suite`: https://github.com/data-apis/array-api-tests
.. _`test_array_api`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/ivy_tests
.. _`for each backend`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/.github/workflows
.. _`array api tests discussion`: https://github.com/unifyai/ivy/discussions/1306
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/ZVQdvbzNQJ
.. _`array api tests channel`: https://discord.com/channels/799879767196958751/982738404611592256

All functions which are present in the `Array API Standard`_ have a corresponding unit test in the
`test suite`_ for the standard.

The test suite is included in the ivy repository as a submodule in the folder `test_array_api`_,
which we keep updated with the upstream test suite.

The entire test suite is run independently `for each backend`_ on every push to the repo.
Therefore, every function which exists in the standard is thoroughly tested for
adherence to the standard on a continuous basis.

You will need to make sure the Array API tests are passing for each backend framework if/when making any changes to Ivy
functions which are part of the standard. If a test fails on the CI, you can see details about the failure under
'Details' -> 'Run [backend] Tests'.

**Round Up**

This should have hopefully given you a good understanding of how the Array API test suite is used for testing Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `array api tests discussion`_,
or reach out on `discord`_ in the `array api tests channel`_!
