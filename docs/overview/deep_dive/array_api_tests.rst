Array API Tests
===============

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`test suite`: https://github.com/data-apis/array-api-tests
.. _`test_array_api`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/ivy_tests
.. _`for each backend`: https://github.com/unifyai/ivy/tree/20d07d7887766bb0d1707afdabe6e88df55f27a5/.github/workflows
.. _`repo`: https://github.com/unifyai/ivy
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`array api tests channel`: https://discord.com/channels/799879767196958751/982738404611592256
.. _`array api tests forum`: https://discord.com/channels/799879767196958751/1028297759738040440
.. _`test_array_api.sh`: https://github.com/unifyai/ivy/blob/d76f0f5ab02d608864eb2c4012af2404da5806c2/test_array_api.sh
.. _`array-api test repository`: https://github.com/data-apis/array-api/tree/main
.. _`issue`: https://github.com/numpy/numpy/issues/21213
.. _`ivy_tests/test_array_api/array_api_tests/test_special_cases.py`: https://github.com/data-apis/array-api-tests/blob/ddd3b7a278cd0c0b68c0e4666b2c9f4e67b7b284/array_api_tests/test_special_cases.py
.. _`here`: https://lets-unify.ai/ivy/contributing/setting_up.html#setting-up-testing
.. _`git website`: https://www.git-scm.com/book/en/v2/Git-Tools-Submodules
.. _`hypothesis`: https://hypothesis.readthedocs.io/en/latest/
.. _`ivy tests`: https://lets-unify.ai/ivy/deep_dive/ivy_tests.html
.. _`final section`: https://lets-unify.ai/ivy/deep_dive/ivy_tests.html#re-running-failed-ivy-tests
.. _`CI Pipeline`: https://lets-unify.ai/ivy/deep_dive/continuous_integration.html#ci-pipeline

In conjunction with our own ivy unit tests, we import the array-api `test suite`_.
These tests check that all ivy backend libraries behave according to the `Array API Standard`_ which was established in May 2020 by a group of maintainers.
It was intended to bring some consistency and completeness to the various python libraries that have gained popularity in the last 5-10 years.
Since Ivy aims to unify machine learning frameworks, it makes sense that we value consistency in behaviour across each of the backend libraries in our code too.

The test suite is included in the ivy repository as a submodule in the folder `test_array_api`_, which we keep updated with the upstream test suite.
The array-api tests repository is maintained by a group of developers unconnected to Ivy.
We have made the decision to import the test suite directly from this repository rather than having our own fork.
This means that the test suite you see in the ivy source code cannot be modified in the usual way of pushing to the ivy master branch.
Instead, the change must be made to the array-api repository directly and then our submodule must be updated with the commands:

.. code-block:: none

        # to initialise local config file and fetch + checkout submodule (not needed everytime)
        git submodule update --init --recursive

        # pulls changes from upstream remote repo and merges them
        git submodule update --recursive --remote --merge

and only *then* can changes to the submodule be pushed to ivy-master, i.e. only when these changes exist in the source array-api repository.
See the `git website`_ for further information on working with submodules.

Running the Tests
-----------------

The entire test suite is run independently `for each backend`_ on every push to the repo.
Therefore, every function which exists in the standard is thoroughly tested for adherence to the standard on a continuous basis.

You will need to make sure the Array API tests are passing for each backend framework if/when making any changes to Ivy functions which are part of the standard.
If a test fails on the CI, you can see details about the failure under `Details -> Run [backend] Tests` as shown in `CI Pipeline`_.

You can also run the tests locally before making a PR.
There are two ways to do this: using the terminal or using your IDE.

Using Terminal
**************

Using the terminal, you can run all array-api tests in a given file for a certain backend using the bash file `test_array_api.sh`_:

.. code-block:: none

        # /ivy
        /bin/bash -e ./run_tests_CLI/test_array_api.sh jax test_linalg

You can change the argument with any of our supported frameworks - tensorflow, numpy, torch or jax - and the individual test function categories in :code:`ivy/ivy_tests/test_array_api/array_api_tests`, e.g. *test_set_functions*, *test_signatures* etc.

You can also run a specific test, as often running *all* tests in a file is excessive.
To make this work, you should set the backend explicitly in the `_array_module.py` file, which you can find in the `array_api_tests` submodule.
At the beginning of the file, you will see the following line of code :code:`array_module = None`.
You need to comment out that line and add the following:

.. code-block:: none

        import ivy as array_module
        array_module.set_backend("jax") # or numpy, tensorflow, torch

You should now be able to run the following commands via terminal:

.. code-block:: none

        # run all tests in a file
        pytest -vv ivy_tests/test_array_api/array_api_tests/test_manipulation_functions.py

        # run a single test
        pytest -vv ivy_tests/test_array_api/array_api_tests/test_manipulation_functions.py -k "test_concat"

Using the IDE
*************

You can also run a specific test or test file by using your IDE.
To make this work, you should set the backend explicitly in the `_array_module.py` file as explained in the previous subsection.
After that, you can run the API test files as you typically would with other tests.
See `here`_  for instructions on how to run tests in ivy more generally.

*NB*: make sure to not add any changes to the array-api files to your commit.

Regenerating Test Failures
--------------------------
Array-API tests are written using `hypothesis`_ to perform property-based testing, just like the `ivy tests`_.
However, unlike the ivy tests, the Array-API tests make liberal use of :code:`data.draw` in the main body of the test function instead of generating the data in the :code:`@given` decorator that wraps it.
This means that failed tests cannot be re-run with the :code:`@example` decorator, as explained in the `final section`_ of the ivy tests deep dive.
Fortunately, it is possible to regenerate test failures using a unique decorator that appears in the final line of the falsifying example in the error stack trace:

.. code-block:: none

    =================================== FAILURES ===================================
    ______________________ test_remainder[remainder(x1, x2)] _______________________
    ivy_tests/test_array_api/array_api_tests/test_operators_and_elementwise_functions.py:1264: in test_remainder
        @given(data=st.data())
    ivy_tests/test_array_api/array_api_tests/test_operators_and_elementwise_functions.py:1277: in test_remainder
        binary_param_assert_against_refimpl(ctx, left, right, res, "%", operator.mod)
    ivy_tests/test_array_api/array_api_tests/test_operators_and_elementwise_functions.py:620: in binary_param_assert_against_refimpl
        binary_assert_against_refimpl(
    ivy_tests/test_array_api/array_api_tests/test_operators_and_elementwise_functions.py:324: in binary_assert_against_refimpl
        assert isclose(scalar_o, expected), (
    E   AssertionError: out=-2.0, but should be roughly (x1 % x2)=1.0 [remainder()]
    E     x1=17304064.0, x2=3.0
    E   assert False
    E    +  where False = isclose(-2.0, 1.0)
    E   Falsifying example: test_remainder(
    E       data=data(...), ctx=BinaryParamContext(<remainder(x1, x2)>),
    E   )
    E   Draw 1 (x1): ivy.array(17304064.)
    E   Draw 2 (x2): ivy.array(3.)
    E
    E   You can reproduce this example by temporarily adding @reproduce_failure('6.55.0', b'AXic42BAAowcnP+RuMwMABAeAR0=') as a decorator on your test case

Copy the :code:`@reproduce_failure` decorator and paste it after the usual decorators of `test_remainder`.
You may also need to include the hypothesis import of `reproduce_failure` as shown below.

.. code-block:: none

    from hypothesis import reproduce_failure

    @pytest.mark.parametrize("ctx", make_binary_params("remainder", dh.numeric_dtypes))
    @given(data=st.data())
    @reproduce_failure('6.55.0', b'AXic42BAAowcnP+RuMwMABAeAR0=')
    def test_remainder(ctx, data):
        left = data.draw(ctx.left_strat, label=ctx.left_sym)
        right = data.draw(ctx.right_strat, label=ctx.right_sym)
        if ctx.right_is_scalar:
            assume(right != 0)
        else:
            assume(not xp.any(right == 0))

        res = ctx.func(left, right)

        binary_param_assert_dtype(ctx, left, right, res)
        binary_param_assert_shape(ctx, left, right, res)
        binary_param_assert_against_refimpl(ctx, left, right, res, "%", operator.mod)

The test should then include the inputs which led to the previous failure and recreate it.
If you are taking the :code:`@reproduce_failure` decorator from a CI stack trace and trying to reproduce it locally, you may find that sometimes the local test unexpectedly passes.
This is usually caused by a discrepancy in your local source code and ivy-master, so try pulling from master to sync the behaviour.

Test Skipping
-------------

Certain tests may need to be skipped when running the array-api test suite.
This could be due to a variety of reasons:

#. the test function has a known issue which the `array-api test repository`_ developers are working on (e.g. :code:`test_asarray_arrays`)
#. the function itself deviates from the standard (e.g. :code:`test_floor_divide`)
#. there is an issue with the hypothesis test data generation i.e. a failed 'health check' (e.g. :code:`test_iop[__imod__(x1_i < 0 and x2_i is +0) -> NaN]`)
#. tolerance issues when asserting output :code:`isequal()` (e.g. :code:`test_matrix_norm`)

All the examples in this list except point 3 (which only occurs with tensorflow) refer to numpy functions, and the first two are skipped in the `array-api test repository`_ also.
The data generation and tolerance issues are not skipped in the array-api repo and are difficult for Ivy developers to solve as we cannot alter the tests directly.
Currently, we import the test suite and run it; we do not have our own fork that we can tweak at will.
These issues have been raised in the array-api test repo and will be addressed in due course.

There are currently two ways to skip array-api tests:

#. in :code:`ivy_tests/array_api_methods_to_test/<submodule>.txt` and
#. in :code:`ivy_tests/skips.txt`

The first method was implemented before the second.
Each :code:`<submodule>.txt` file contains a comprehensive list of functions which belong to that submodule, some of which are commented out.
The commented-out functions are being skipped *only* for the backend(s) that is/are causing the failure, not all the backends.
This is done by identifying any references to a backend in the commented-out line e.g. :code:`#trace # failing for jax, numpy due to issues with dtypes in output in test: https://github.com/data-apis/array-api/issues/202` will cause :code:`test_trace` to be skipped on the jax and numpy backends.

The latter method, on the other hand, skips a test on *all* backends, even if it is just failing on one.
The :code:`ivy_tests/skips.txt` scheme was implemented to skip *specific test cases*.
The array-api test suite contains a set of special tests which aim to cover edge-case input and particular data type promotion rules (see :code:`ivy_tests/test_array_api/array_api_tests/test_special_cases.py`).
In :code:`ivy_tests/skips.txt`, tests are skipped by writing the filepath + conditions on the input of the test e.g.,

.. code-block:: bash

    ivy_tests/test_array_api/array_api_tests/test_special_cases.py::test_iop[__ipow__(x1_i is -infinity and x2_i > 0 and not (x2_i.is_integer() and x2_i % 2 == 1)) -> +infinity]

is skipping the in-place operations test on the :code:`pow` instance method when x1 is -infinity and x2 is a positive, odd float.
The result should be +infinity, however there is a known problem with the numpy instance method and an `issue`_ has been raised on the numpy repository.
Tests are categorised in :code:`ivy_tests/skips.txt` according to the backend they are failing on and the reason for the failure.
The fact that the skip instruction itself contains the exact input conditions that are failing makes it easier to keep track of and revisit failing tests to try and fix them.

**Round Up**

This should have hopefully given you a good understanding of how the Array API test suite is used for testing Ivy.

If you have any questions, please feel free to reach out on `discord`_ in the `array api tests channel`_ or in the `array api tests forum`_ !

**Video**

.. raw:: html

    <iframe width="420" height="315"
    src="https://www.youtube.com/embed/R1XuYwzhxWw" class="video">
    </iframe>
