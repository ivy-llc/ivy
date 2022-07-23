import inspect
from typing import Union
from hypothesis import strategies as st
from hypothesis import given


def bool_val_flags(cl_arg: Union[bool, None]):
    if cl_arg is not None:
        return st.booleans().filter(lambda x: x == cl_arg)
    return st.booleans()


def _handle_var(test_fn):
    def new_fn(data, get_command_line_flags, **kwargs):

        for param in inspect.signature(test_fn).parameters.values():
            if param.kind == param.KEYWORD_ONLY:
                if param.name == "as_variable":
                    as_variable = data.draw(
                        bool_val_flags(get_command_line_flags["as-variable"])
                    )
                    kwargs["as_variable"] = as_variable
                if param.name == "native_array":
                    native_array = data.draw(
                        bool_val_flags(get_command_line_flags["native-array"])
                    )
                    kwargs["native_array"] = native_array
                if param.name == "with_out":
                    with_out = data.draw(
                        bool_val_flags(get_command_line_flags["with-out"])
                    )
                    kwargs["with_out"] = with_out
                if param.name == "instance_method":
                    instance_method = data.draw(
                        bool_val_flags(get_command_line_flags["instance-method"])
                    )
                    kwargs["instance_method"] = instance_method
                if param.name == "container":
                    container = data.draw(
                        bool_val_flags(get_command_line_flags["nestable"])
                    )
                    kwargs["container"] = container

        return test_fn(**kwargs)

    return new_fn


@given(data=st.data())
@_handle_var
def test_lol(*, as_variable, native_array, with_out, container, instance_method):
    assert as_variable == False
    assert native_array == True
    assert with_out == False
    assert container == True
    assert instance_method == False
