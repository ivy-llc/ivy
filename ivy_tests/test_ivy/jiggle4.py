from typing import Union
from hypothesis import strategies as st
from hypothesis import given


def bool_val_flags(cl_arg: Union[bool, None]):
    if cl_arg is not None:
        return st.booleans().filter(lambda x: x == cl_arg)
    return st.booleans()


@given(data=st.data())
def test_lol(get_command_line_flags, data):
    as_variable = data.draw(bool_val_flags(get_command_line_flags["as-variable"]))
    assert as_variable == True
