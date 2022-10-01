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
.. _`array-api test repository`: https://github.com/data-apis/array-api/tree/main
.. _`issue`: https://github.com/numpy/numpy/issues/21213
.. _`ivy_tests/test_array_api/array_api_tests/test_special_cases.py`: https://github.com/data-apis/array-api-tests/blob/ddd3b7a278cd0c0b68c0e4666b2c9f4e67b7b284/array_api_tests/test_special_cases.py

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

Test Skipping
-------------
Certain tests may need to be skipped when running the array-api test suite. This could be due to a variety of reasons:

#. the test function has a known issue which the `array-api test repository`_ developers are working on (e.g. :code:`test_asarray_arrays`)
#. the function itself deviates from the standard (e.g. :code:`test_floor_divide`)
#. there is an issue with the hypothesis test data generation i.e. a failed 'health check' (e.g. :code:`test_iop[__imod__(x1_i < 0 and x2_i is +0) -> NaN]`)
#. tolerance issues when asserting output :code:`isequal()` (e.g. :code:`test_matrix_norm`)

All the examples in this list except point 3 (which only occurs with tensorflow) refer to numpy functions, and the first
two are skipped in the `array-api test repository`_ also. The data generation and tolerance issues are not skipped in the
array-api repo and are difficult for Ivy developers
to solve as we cannot alter the tests directly. Currently, we import the test suite and run it; we do not
have our own fork that we can tweak at will. The rationale for doing so is that we should adhere as closely to the
standard as possible. These issues have been raised in the array-api test repo and will be addressed in due course.

There are currently two ways to skip array-api tests:

#. in :code:`ivy_tests/array_api_methods_to_test/<submodule>.txt` and
#. in :code:`ivy_tests/skips.txt`

The first method was implemented before the second. Each :code:`<submodule>.txt` file contains a comprehensive list
of functions which belong to that submodule, some of which are commented out. The commented-out functions are being
skipped *only* for the backend(s) that is/are causing the failure, not all the backends. The latter method, on the
other hand, skips a test on *all* backends, even if
it is just failing on one. The :code:`ivy_tests/skips.txt` scheme was implemented to skip *specific test cases*. The array-api
test suite contains a set of special tests which aim to cover edge-case input and particular data type promotion rules
(see :code:`ivy_tests/test_array_api/array_api_tests/test_special_cases.py`). In :code:`ivy_tests/skips.txt`, tests are
skipped by writing the filepath + conditions on the input of the test e.g.,

.. code-block:: bash

    ivy_tests/test_array_api/array_api_tests/test_special_cases.py::test_iop[__ipow__(x1_i is -infinity and x2_i > 0 and not (x2_i.is_integer() and x2_i % 2 == 1)) -> +infinity]

is skipping the in-place operations test on the :code:`pow`
instance method when x1 is -infinity and x2 is a positive, odd float. The result should be +infinity, however there is
an issue with the numpy instance method and an `issue`_ has been raised on the numpy repository. Tests are categorised
in :code:`ivy_tests/skips.txt` according to the backend they are failing on and the reason for the failure. This should
make unskipping temporarily failing tests straightforward once the issue has been resolved, especially if the skip instruction
itself contains the exact input conditions that are failing.

**Round Up**

This should have hopefully given you a good understanding of how the Array API test suite is used for testing Ivy.

If you're ever unsure of how best to proceed,
please feel free to engage with the `array api tests discussion`_,
or reach out on `discord`_ in the `array api tests channel`_!
