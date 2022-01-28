# Test Suite for Array API Compliance

This is the test suite for array libraries adopting the [Python Array API
standard](https://data-apis.org/array-api/latest).

Note the suite is still a **work in progress**. Feedback and contributions are
welcome!

## Quickstart

### Setup

To run the tests, install the testing dependencies.

```bash
$ pip install -r requirements.txt
```

Ensure you have the array library that you want to test installed.

### Specifying the array module

You need to specify the array library to test. It can be specified via the
`ARRAY_API_TESTS_MODULE` environment variable, e.g.

```bash
$ export ARRAY_API_TESTS_MODULE=numpy.array_api
```

Alternately, change the `array_module` variable in `array_api_tests/_array_module.py`
line, e.g.

```diff
- array_module = None
+ import numpy.array_api as array_module
```

### Run the suite

Simply run `pytest` against the `array_api_tests/` folder to run the full suite.

```bash
$ pytest array_api_tests/
```

The suite tries to logically organise its tests. `pytest` allows you to only run
a specific test case, which is useful when developing functions.

```bash
$ pytest array_api_tests/test_creation_functions.py::test_zeros
```

## Releases

The test suite has tagged releases on
[GitHub](https://github.com/data-apis/array-api-tests/releases). If you run
the test suite in your CI system, we recommend pinning against a release tag.

We use [calender versioning](https://calver.org/) for the releases. You should
expect that any version may be "breaking" compared to the previous one, in the
sense that there may have been additional tests added which cause a previously
passing library to fail.

For now, the test suite is
not installable as a Python package. You can use it by cloning the repo and
running `pytest` as described above. If it would help you to be able to
install it as a package, [please let us
know](https://github.com/data-apis/array-api-tests/issues/85).

*Test suite maintainer note:* to make a release of the test suite, make an
annotated tag with the version:

```
git tag -a 2022.1
```

(for the message, just write something like "array-api-tests version 2022.1").
Be sure to use the calver version number for the tag name. Versioneer will
automatically set the version number of the `array_api_tests` package based on
the git tag.

Then push the tag to GitHub

```
git push --tags origin 2022.1
```

Finally go to the [tags page on
GitHub](https://github.com/data-apis/array-api-tests/tags) and convert the tag
into a release. If you want, you can add release notes to the release page on
GitHub.


## What the test suite covers

We are interested in array libraries conforming to the
[spec](https://data-apis.org/array-api/latest/API_specification/index.html).
Ideally this means that if a library has fully adopted the Array API, the test
suite passes. We take great care to _not_ test things which are out-of-scope,
so as to not unexpectedly fail the suite.

### Primary tests

Every function—including array object methods—has a respective test method. We
use [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) to generate a
diverse set of valid inputs. This means array inputs will cover different dtypes
and shapes, as well as contain interesting elements. These examples generate
with interesting arrangements of non-array positional arguments and keyword
arguments.

Each test case will cover the following areas if relevant:

* **Smoking**: We pass our generated examples to all functions. As these
  examples solely consist of *valid* inputs, we are testing that functions can
  be called using their documented inputs without raising errors.

* **Data type**: For functions returning/modifying arrays, we assert that output
  arrays have the correct data types. Most functions
  [type-promote](https://data-apis.org/array-api/latest/API_specification/type_promotion.html)
  input arrays and some functions have bespoke rules—in both cases we simulate
  the correct behaviour to find the expected data types.

* **Shape**: For functions returning/modifying arrays, we assert that output
  arrays have the correct shape. Most functions
  [broadcast](https://data-apis.org/array-api/latest/API_specification/broadcasting.html)
  input arrays and some functions have bespoke rules—in both cases we simulate
  the correct behaviour to find the expected shapes.

* **Values**: We assert output values (including the elements of
  returned/modified arrays) are as expected. Except for manipulation functions
  or special cases, the spec allows floating-point inputs to have inexact
  outputs, so with such examples we only assert values are roughly as expected.

### Additional tests

In addition to having one test case for each function, we test other properties
of the functions and some miscellaneous things.

* **Special cases**: For functions with special case behaviour, we assert that
  these functions return the correct values.

* **Signatures**: We assert functions have the correct signatures.

* **Constants**: We assert that
  [constants](https://data-apis.org/array-api/latest/API_specification/constants.html)
  behave expectedly, are roughly the expected value, and that any related
  functions interact with them correctly.

Be aware that some aspects of the spec are impractical or impossible to actually
test, so they are not covered in the suite <!-- TODO: note what these are -->

## Interpreting errors

First and foremost, note that most tests have to assume that certain aspects of
the Array API have been correctly adopted, as fundamental APIs such as array
creation and equalities are hard requirements for many assertions. This means a
test case for one function might fail because another function has bugs or even
no implementation.

This means adopting libraries at first will result in a vast number of errors
due to cascading errors. Generally the nature of the spec means many granular
details such as type promotion is likely going to also fail nearly-conforming
functions.

We hope to improve user experience in regards to "noisy" errors in
[#51](https://github.com/data-apis/array-api-tests/issues/51). For now, if an
error message involves `_UndefinedStub`, it means an attribute of the array
library (including functions) and it's objects (e.g. the array) is missing.

The spec is the suite's source of truth. If the suite appears to assume
behaviour different from the spec, or test something that is not documented,
this is a bug—please [report such
issues](https://github.com/data-apis/array-api-tests/issues/) to us.

## Configuration

By default, tests for the optional Array API extensions such as
[`linalg`](https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html)
will be skipped if not present in the specified array module. You can purposely
skip testing extension(s) via the `--disable-extension` option.

The tests make heavy use
[Hypothesis](https://hypothesis.readthedocs.io/en/latest/). You can configure
how many examples are generated using the `--max-examples` flag, which defaults
to 100. Lower values can be useful for quick checks, and larger values should
result in more rigorous runs. For example, `--max-examples 10000` may find bugs
where default runs don't, but will take a much longer time.

<!-- TODO: howto on CI -->

## Contributing

### Remain in-scope

It is important that every test only uses APIs that are part of the standard.
For instance, when creating input arrays you should only use the [array creation
functions](https://data-apis.org/array-api/latest/API_specification/creation_functions.html)
that are documented in the spec. The same goes for testing arrays—you'll find
many utilities that parralel NumPy's own test utils in the `*_helpers.py` files.

### Tools

Hypothesis should always be used for the primary tests, and can be useful
elsewhere. Effort should be made so drawn arguments are labeled with their
respective names. For
[`st.data()`](https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.data),
draws should be accompanied with the `label` kwarg i.e. `data.draw(<strategy>,
label=<label>)`.

[`pytest.mark.parametrize`](https://docs.pytest.org/en/latest/how-to/parametrize.html)
should be used to run tests over multiple arguments. Parameterization should be
preferred over using Hypothesis when there are a small number of possible
inputs, as this allows better failure reporting. Note using both parametrize and
Hypothesis for a single test method is possible and can be quite useful.

### Error messages

Any assertion should be accompanied with a descriptive error message, including
the relevant values. Error messages should be self-explanatory as to why a given
test fails, as one should not need prior knowledge of how the test is
implemented.

### Generated files

Some files in the suite are automatically generated from the spec, and should
not be edited directly. To regenerate these files, run the script

    ./generate_stubs.py path/to/array-api

where `path/to/array-api` is the path to a local clone of the [`array-api`
repo](https://github.com/data-apis/array-api/). Edit `generate_stubs.py` to make
changes to the generated files.

## Future plans

Keeping full coverage of the spec is an on-going priority as the Array API
evolves.

Additionally, we have features and general improvements planned. Work on such
functionality is guided primarily by the concerete needs of developers
implementing and using the Array API—be sure to [let us
know](https://github.com/data-apis/array-api-tests/issues) any limitations you
come across.

* A dependency graph for every test case, which could be used to modify pytest's
  collection so that low-dependency tests are run first, and tests with faulty
  dependencies would skip/xfail.

* In some tests we've found it difficult to find appropaite assertion parameters
  for output values (particularly epsilons for floating-point outputs), so we
  need to review these and either implement assertions or properly note the lack
  thereof.
