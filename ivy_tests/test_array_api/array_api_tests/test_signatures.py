import inspect

import pytest

from ._array_module import mod, mod_name, ones, eye, float64, bool, int64, _UndefinedStub
from .pytest_helpers import raises, doesnt_raise
from . import dtype_helpers as dh

from . import function_stubs


submodules = [m for m in dir(function_stubs) if
              inspect.ismodule(getattr(function_stubs, m)) and not
              m.startswith('_')]

def stub_module(name):
    for m in submodules:
        if name in getattr(function_stubs, m).__all__:
            return m

def extension_module(name):
    return name in submodules and name in function_stubs.__all__

extension_module_names = []
for n in function_stubs.__all__:
    if extension_module(n):
        extension_module_names.extend([f'{n}.{i}' for i in getattr(function_stubs, n).__all__])


params = []
for name in function_stubs.__all__:
    marks = []
    if extension_module(name):
        marks.append(pytest.mark.xp_extension(name))
    params.append(pytest.param(name, marks=marks))
for name in extension_module_names:
    ext = name.split('.')[0]
    mark = pytest.mark.xp_extension(ext)
    params.append(pytest.param(name, marks=[mark]))


def array_method(name):
    return stub_module(name) == 'array_object'

def function_category(name):
    return stub_module(name).rsplit('_', 1)[0].replace('_', ' ')

def example_argument(arg, func_name, dtype):
    """
    Get an example argument for the argument arg for the function func_name

    The full tests for function behavior is in other files. We just need to
    have an example input for each argument name that should work so that we
    can check if the argument is implemented at all.

    """
    # Note: for keyword arguments that have a default, this should be
    # different from the default, as the default argument is tested separately
    # (it can have the same behavior as the default, just not literally the
    # same value).
    known_args = dict(
        api_version='2021.1',
        arrays=(ones((1, 3, 3), dtype=dtype), ones((1, 3, 3), dtype=dtype)),
        # These cannot be the same as each other, which is why all our test
        # arrays have to have at least 3 dimensions.
        axis1=2,
        axis2=2,
        axis=1,
        axes=(2, 1, 0),
        copy=True,
        correction=1.0,
        descending=True,
        # TODO: This will only work on the NumPy implementation. The exact
        # value of the device keyword will vary across implementations, so we
        # need some way to infer it or for libraries to specify a list of
        # valid devices.
        device='cpu',
        dtype=float64,
        endpoint=False,
        fill_value=1.0,
        from_=int64,
        full_matrices=False,
        k=1,
        keepdims=True,
        key=(0, 0),
        indexing='ij',
        mode='complete',
        n=2,
        n_cols=1,
        n_rows=1,
        num=2,
        offset=1,
        ord=1,
        obj = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
        other=ones((3, 3), dtype=dtype),
        return_counts=True,
        return_index=True,
        return_inverse=True,
        rtol=1e-10,
        self=ones((3, 3), dtype=dtype),
        shape=(1, 3, 3),
        shift=1,
        sorted=False,
        stable=False,
        start=0,
        step=2,
        stop=1,
        # TODO: Update this to be non-default. See the comment on "device" above.
        stream=None,
        to=float64,
        type=float64,
        upper=True,
        value=0,
        x1=ones((1, 3, 3), dtype=dtype),
        x2=ones((1, 3, 3), dtype=dtype),
        x=ones((1, 3, 3), dtype=dtype),
    )
    if not isinstance(bool, _UndefinedStub):
        known_args['condition'] = ones((1, 3, 3), dtype=bool),

    if arg in known_args:
        # Special cases:

        # squeeze() requires an axis of size 1, but other functions such as
        # cross() require axes of size >1
        if func_name == 'squeeze' and arg == 'axis':
            return 0
        # ones() is not invertible
        # finfo requires a float dtype and iinfo requires an int dtype
        elif func_name == 'iinfo' and arg == 'type':
            return int64
        # tensordot args must be contractible with each other
        elif func_name == 'tensordot' and arg == 'x2':
            return ones((3, 3, 1), dtype=dtype)
        # tensordot "axes" is either a number representing the number of
        # contractible axes or a 2-tuple or axes
        elif func_name == 'tensordot' and arg == 'axes':
            return 1
        # The inputs to outer() must be 1-dimensional
        elif func_name == 'outer' and arg in ['x1', 'x2']:
            return ones((3,), dtype=dtype)
        # Linear algebra functions tend to error if the input isn't "nice" as
        # a matrix
        elif arg.startswith('x') and func_name in function_stubs.linalg.__all__:
            return eye(3)
        return known_args[arg]
    else:
        raise RuntimeError(f"Don't know how to test argument {arg}. Please update test_signatures.py")

@pytest.mark.parametrize('name', params)
def test_has_names(name):
    if extension_module(name):
        assert hasattr(mod, name), f'{mod_name} is missing the {name} extension'
    elif '.' in name:
        extension_mod, name = name.split('.')
        assert hasattr(getattr(mod, extension_mod), name), f"{mod_name} is missing the {function_category(name)} extension function {name}()"
    elif array_method(name):
        arr = ones((1, 1))
        if getattr(function_stubs.array_object, name) is None:
            assert hasattr(arr, name), f"The array object is missing the attribute {name}"
        else:
            assert hasattr(arr, name), f"The array object is missing the method {name}()"
    else:
        assert hasattr(mod, name), f"{mod_name} is missing the {function_category(name)} function {name}()"

@pytest.mark.parametrize('name', params)
def test_function_positional_args(name):
    # Note: We can't actually test that positional arguments are
    # positional-only, as that would require knowing the argument name and
    # checking that it can't be used as a keyword argument. But argument name
    # inspection does not work for most array library functions that are not
    # written in pure Python (e.g., it won't work for numpy ufuncs).

    if extension_module(name):
        return

    dtype = None
    if (name.startswith('__i') and name not in ['__int__', '__invert__', '__index__']
        or name.startswith('__r') and name != '__rshift__'):
        n = f'__{name[3:]}'
    else:
        n = name
    in_dtypes = dh.func_in_dtypes.get(n, dh.float_dtypes)
    if bool in in_dtypes:
        dtype = bool
    elif all(d in in_dtypes for d in dh.all_int_dtypes):
        dtype = int64

    if array_method(name):
        if name == '__bool__':
            _mod = ones((), dtype=bool)
        elif name in ['__int__', '__index__']:
            _mod = ones((), dtype=int64)
        elif name == '__float__':
            _mod = ones((), dtype=float64)
        else:
            _mod = example_argument('self', name, dtype)
        stub_func = getattr(function_stubs, name)
    elif '.' in name:
        extension_module_name, name = name.split('.')
        _mod = getattr(mod, extension_module_name)
        stub_func = getattr(getattr(function_stubs, extension_module_name), name)
    else:
        _mod = mod
        stub_func = getattr(function_stubs, name)

    if not hasattr(_mod, name):
        pytest.skip(f"{mod_name} does not have {name}(), skipping.")
    if stub_func is None:
        # TODO: Can we make this skip the parameterization entirely?
        pytest.skip(f"{name} is not a function, skipping.")
    mod_func = getattr(_mod, name)
    argspec = inspect.getfullargspec(stub_func)
    func_args = argspec.args
    if func_args[:1] == ['self']:
        func_args = func_args[1:]
    nargs = [len(func_args)]
    if argspec.defaults:
        # The actual default values are checked in the specific tests
        nargs.extend([len(func_args) - i for i in range(1, len(argspec.defaults) + 1)])

    args = [example_argument(arg, name, dtype) for arg in func_args]
    if not args:
        args = [example_argument('x', name, dtype)]
    else:
        # Duplicate the last positional argument for the n+1 test.
        args = args + [args[-1]]

    kwonlydefaults = argspec.kwonlydefaults or {}
    required_kwargs = {arg: example_argument(arg, name, dtype) for arg in argspec.kwonlyargs if arg not in kwonlydefaults}

    for n in range(nargs[0]+2):
        if name == 'result_type' and n == 0:
            # This case is not encoded in the signature, but isn't allowed.
            continue
        if n in nargs:
            doesnt_raise(lambda: mod_func(*args[:n], **required_kwargs))
        elif argspec.varargs:
            pass
        else:
            # NumPy ufuncs raise ValueError instead of TypeError
            raises((TypeError, ValueError), lambda: mod_func(*args[:n]), f"{name}() should not accept {n} positional arguments")

@pytest.mark.parametrize('name', params)
def test_function_keyword_only_args(name):
    if extension_module(name):
        return

    if array_method(name):
        _mod = ones((1, 1))
        stub_func = getattr(function_stubs, name)
    elif '.' in name:
        extension_module_name, name = name.split('.')
        _mod = getattr(mod, extension_module_name)
        stub_func = getattr(getattr(function_stubs, extension_module_name), name)
    else:
        _mod = mod
        stub_func = getattr(function_stubs, name)

    if not hasattr(_mod, name):
        pytest.skip(f"{mod_name} does not have {name}(), skipping.")
    if stub_func is None:
        # TODO: Can we make this skip the parameterization entirely?
        pytest.skip(f"{name} is not a function, skipping.")
    mod_func = getattr(_mod, name)
    argspec = inspect.getfullargspec(stub_func)
    args = argspec.args
    if args[:1] == ['self']:
        args = args[1:]
    kwonlyargs = argspec.kwonlyargs
    kwonlydefaults = argspec.kwonlydefaults or {}
    dtype = None

    args = [example_argument(arg, name, dtype) for arg in args]

    for arg in kwonlyargs:
        value = example_argument(arg, name, dtype)
        # The "only" part of keyword-only is tested by the positional test above.
        doesnt_raise(lambda: mod_func(*args, **{arg: value}),
                     f"{name}() should accept the keyword-only argument {arg!r}")

        # Make sure the default is accepted. These tests are not granular
        # enough to test that the default is actually the default, i.e., gives
        # the same value if the keyword isn't passed. That is tested in the
        # specific function tests.
        if arg in kwonlydefaults:
            default_value = kwonlydefaults[arg]
            doesnt_raise(lambda: mod_func(*args, **{arg: default_value}),
                         f"{name}() should accept the default value {default_value!r} for the keyword-only argument {arg!r}")
