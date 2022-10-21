# general
import importlib
import inspect
from hypothesis import given, strategies as st

# local
import ivy
from ivy_tests.test_ivy import conftest as cfg  # TODO temporary
from .hypothesis_helpers import number_helpers as nh
from .globals import TestData


cmd_line_args = (
    "with_out",
    "instance_method",
    "test_gradients",
)
cmd_line_args_lists = (
    "as_variable",
    "native_array",
    "container",
)


@st.composite
def num_positional_args(draw, *, fn_name: str = None):
    """Draws an integers randomly from the minimum and maximum number of positional
    arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    fn_name
        name of the function.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.

    Examples
    --------
    @given(
        num_positional_args=num_positional_args(fn_name="floor_divide")
    )
    @given(
        num_positional_args=num_positional_args(fn_name="add")
    )
    """
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    fn = None
    for i, fn_name_key in enumerate(fn_name.split(".")):
        if i == 0:
            fn = ivy.__dict__[fn_name_key]
        else:
            fn = fn.__dict__[fn_name_key]
    for param in inspect.signature(fn).parameters.values():
        if param.name == "self":
            continue
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
            num_keyword_only += 1
        elif param.kind == param.VAR_KEYWORD:
            num_keyword_only += 1
    return draw(
        nh.ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


@st.composite
def num_positional_args_from_fn(draw, *, fn: str = None):
    """Draws an integers randomly from the minimum and maximum number of positional
    arguments a given function can take.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    fn
        name of the function.

    Returns
    -------
    A strategy that can be used in the @given hypothesis decorator.

    Examples
    --------
    @given(
        num_positional_args=num_positional_args_from_fn(fn="floor_divide")
    )
    @given(
        num_positional_args=num_positional_args_from_fn(fn="add")
    )
    """
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    for param in inspect.signature(fn).parameters.values():
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
            num_keyword_only += 1
        elif param.kind == param.VAR_KEYWORD:
            num_keyword_only += 1
    return draw(
        nh.ints(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )


# Decorators


def handle_test(*, fn_tree: str, **_given_kwargs):
    def test_wrapper(test_func):
        def wrapped_test(*args, **kwargs):
            return test_func(*args, **kwargs)

        return wrapped_test

    return test_wrapper


def handle_frontend_test(*, fn_tree: str, **_given_kwargs):
    split_index = fn_tree.rfind(".")
    fn_name = fn_tree[split_index + 1 :]
    module_to_import = fn_tree[:split_index]
    tmp_mod = importlib.import_module(module_to_import)
    callable_fn = tmp_mod.__dict__[fn_name]

    _given_kwargs["num_positional_args"] = num_positional_args(fn_name=fn_tree)
    for flag_key, flag_value in cfg.GENERAL_CONFIG_DICT.items():
        _given_kwargs[flag_key] = st.just(flag_value)
    for flag in cfg.UNSET_TEST_CONFIG:
        _given_kwargs[flag] = st.booleans()

    # Override with_out to be compatible
    # TODO this actually override GENERAL_CONFIG_DICT, should handle this
    for k in inspect.signature(callable_fn).parameters.keys():
        if k.endswith("out"):
            _given_kwargs["with_out"] = st.booleans()
            break
    else:
        _given_kwargs["with_out"] = st.just(False)

    def test_wrapper(test_fn):
        def wrapped_test(fixt_frontend_str, *args, **kwargs):
            __tracebackhide__ = True
            fn_name = wrapped_test.test_data.fn_name
            frontend = fixt_frontend_str
            wrapped_hypothesis_test = given(**_given_kwargs)(test_fn)
            return wrapped_hypothesis_test(
                fn_tree=fn_name, frontend=frontend, *args, **kwargs
            )

        wrapped_test.test_data = TestData(
            test_fn=wrapped_test,
            callable_fn=callable_fn,
            fn_tree=fn_tree,
            fn_name=fn_name,
            unsupported_dtypes=None,  # TODO
        )

        return wrapped_test

    return test_wrapper


@st.composite
def seed(draw):
    return draw(st.integers(min_value=0, max_value=2**8 - 1))
