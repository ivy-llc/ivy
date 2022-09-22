# Verification - test suite

## Measuring conformance

In addition to the specification documents, a test suite is being developed to
aid library developers check conformance to the spec. **NOTE: The test suite
is still a work in progress.** It can be found at
<https://github.com/data-apis/array-api-tests>.

It is important to note that while the aim of the array API test suite is to
cover as much of the spec as possible, there are necessarily some aspects of
the spec that are not covered by the test suite, typically because they are
impossible to effectively test. Furthermore, if the test suite appears to
diverge in any way from what the spec documents say, this should be considered
a bug in the test suite. The specification is the ground source of truth.

## Running the tests

To run the tests, first clone the [test suite
repo](https://github.com/data-apis/array-api-tests), and install the testing
dependencies,

    pip install pytest hypothesis

or

    conda install pytest hypothesis

as well as the array libraries that you want to test. To run the tests, you
need to specify the array library that is to be tested. There are two ways to
do this. One way is to set the `ARRAY_API_TESTS_MODULE` environment variable.
For example

    ARRAY_API_TESTS_MODULE=numpy pytest

Alternatively, edit the `array_api_tests/_array_module.py` file and change the
line

```py
array_module = None
```

to

```py
import numpy as array_module
```

(replacing `numpy` with the array module namespace to be tested).

In either case, the tests should be run with the `pytest` command.

Aside from the two testing dependencies (`pytest` and `hypothesis`), the test
suite has no dependencies. In particular, it does not depend on any specific
array libraries such as NumPy. All tests are run using only the array library
that is being tested, comparing results against the behavior as defined in the
spec. The test suite is designed to be standalone so that it can easily be vendored.

See the
[README](https://github.com/data-apis/array-api-tests/blob/master/README.md)
in the test suite repo for more information about how to run and interpret the
test suite results.
