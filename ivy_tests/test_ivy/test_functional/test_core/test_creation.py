
#global
import pytest

#local
import ivy
from ivy_tests.test_ivy import helpers


def test_zeros():
    # docstring test
    helpers.assert_docstring_examples_run(ivy.zeros)


def test_ones():
    # docstring test
    helpers.assert_docstring_examples_run(ivy.ones)


def test_ones_like():
    # docstring test
    helpers.assert_docstring_examples_run(ivy.ones_like)