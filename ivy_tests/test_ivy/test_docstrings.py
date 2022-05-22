# global
import warnings

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


def test_docstrings():
    failures = list()
    success = True
    for k, v in ivy.__dict__.copy().items():
        if (
            k in ["namedtuple", "DType", "Dtype"]
            or (ivy.current_framework_str() in ["jax", "numpy"] and k == "conv3d")
            or helpers.docstring_examples_run(v)
        ):
            continue
        success = False
        failures.append(k)
    if not success:
        warnings.warn(
            "the following methods had failing docstrings:\n\n{}".format(
                "\n".join(failures)
            )
        )
