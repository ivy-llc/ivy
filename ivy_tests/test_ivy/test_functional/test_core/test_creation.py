from ... import helpers
import ivy


def test_eye():
    # docstring test
    assert helpers.docstring_examples_run(ivy.eye) == True


def test_logspace():
    # docstring test
    assert helpers.docstring_examples_run(ivy.logspace) == True
