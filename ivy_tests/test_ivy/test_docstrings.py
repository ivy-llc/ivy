# local
import ivy
import ivy_tests.test_ivy.helpers as helpers

def test_docstrings():
    failures = list()
    success = True
    for k, v in ivy.__dict__.items():
        if k in ['namedtuple']:
            continue
        res = helpers.docstring_examples_run(v)
        if not res:
            success = False
            failures.append(k)
    if not success:
        raise Exception('the following methods had failing docstrings:\n\n{}'.format('\n'.join(failures)))
