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
.. _`test_array_api.sh`: https://github.com/unifyai/ivy/blob/d76f0f5ab02d608864eb2c4012af2404da5806c2/test_array_api.sh

All functions which are present in the `Array API Standard`_ have a corresponding unit test in the
`test suite`_ for the standard.

The test suite is included in the ivy repository as a submodule in the folder `test_array_api`_,
which we keep updated with the upstream test suite.

Running the tests
-----------------

The entire test suite is run independently `for each backend`_ on every push to the repo.
Therefore, every function which exists in the standard is thoroughly tested for
adherence to the standard on a continuous basis.

You will need to make sure the Array API tests are passing for each backend framework if/when making any changes to Ivy
functions which are part of the standard. If a test fails on the CI, you can see details about the failure under
'Details' -> 'Run [backend] Tests'.

You can also run the tests locally before making a PR. There are two ways to do this: by the terminal or using your IDE.

Using bash file
****

Using the terminal, you can run our bash file `test_array_api.sh`_ and specify which framework backend you want to use.
You can use the following command as an example.

.. code-block:: none

        /bin/bash -e ./test_array_api.sh  '<insert_chosen_backend>'

You can change the argument with any of our other supported frameworks like 'tensorflow' or 'numpy'.

Using IDE
****
If you prefer, you can also run a specific test or test file by using your IDE. To make this work, you should set the
backend explicitly in the '_array_module.py' file. You can find it on the 'array_api_tests' submodule. At the beginning
of the file, you will see the following line of code :code:`array_module = None`. You need to comment that line and add
the following code:

.. code-block:: none

        import ivy as array_module
        array_module.set_backend("<insert_chosen_backend>")

After that, you can run the API test files as you typically would with other tests. Just make sure to not add these
changes to your commit.

Re-Running Failed Array API Tests
****

When a hypothesis test fails, the falsifying example is printed on the console by Hypothesis.
For example, in the :code:`test_trace` Array API Test, we find the following output on running the test:

.. code-block::

        Falsifying example: test_trace(
            x=ivy.array([[1.e-05]]), kw={},
        )

It is always efficient to fix this particular example first, before running any other examples.
In order to achieve this functionality, we can use the :code:`@example` Hypothesis decorator.
The :code:`@example` decorator ensures that a specific example is always tested, on running a particular test.
The decorator requires the test arguments as parameters.
For the :code:`test_trace` Array API Test, we can add the decorator as follows:

.. code-block::

        @example(x=ivy.array([[3.5e-46]]), kw={})
        def test_trace(x, kw):

This ensures that the given example is always tested while running the test, allowing one to debug the failure
efficiently.

**Round Up**

This should have hopefully given you a good understanding of how the Array API test suite is used for testing Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `array api tests discussion`_,
or reach out on `discord`_ in the `array api tests channel`_!
