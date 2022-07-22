import inspect
from typing import Union
from hypothesis import strategies as st
from hypothesis import given


def bool_val_flags(cl_arg: Union[bool, None]):
    if cl_arg is not None:
        return st.booleans().filter(lambda x: x == cl_arg)
    return st.booleans()

def _handle_var(test_fn):
    def new_fn(data, get_command_line_flags, *args, **kwargs):

        # if test_fn has keyword argument as_variable then:

        #---------  should all be indented in an if statement
        as_variable = data.draw(bool_val_flags(get_command_line_flags["as-variable"]))
        kwargs["as_variable"] = as_variable
        #----------

        # etc.

        return test_fn(*args, **kwargs)
    return new_fn


@given(data=st.data())
@_handle_var
def test_lol(as_variable):
    assert as_variable == True
