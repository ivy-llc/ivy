import ivy
from ivy_tests.test_docstrings import check_docstring_examples_run

backend = "numpy"
ivy.set_backend(backend)
d = ivy.__dict__
fn = "trace"
method = d[fn]
print(check_docstring_examples_run(fn=method))
