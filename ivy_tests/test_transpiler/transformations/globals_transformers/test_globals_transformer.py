import pytest
import gast
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.globals_transformer.base_transformer import (
    BaseGlobalsTransformer,
)
from ivy.transpiler.transformations.transformers.canonicalize_transformer.base_transformer import (
    BaseNameCanonicalizer,
)
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import ConfigurationsContainer
from ivy.transpiler.utils.ast_utils import ast_to_source_code
from ivy.transpiler import transpile
from ivy_tests.test_transpiler.kornia.helpers import (
    _nest_torch_tensor_to_new_framework,
    _check_allclose,
    _nest_array_to_numpy,
)
import torch


class MockGlobalsTransformer(BaseGlobalsTransformer):
    """
    A simplified version of the BaseGlobalsTransformer that doesnt recursively
    transform the RHS of the globals assignment. This is only useful for testing purposes
    and shouldnt be used in practice.
    """

    def transform_node(self, node, module):
        """variant of the transform method to apply transformations on a targeted node."""
        # dont transform the RHS of the node
        return node


def transform_function(func, apply_canonicalization=False):
    container = ConfigurationsContainer()
    container.load_configurations(source="torch", target="torch_frontend")
    object_like = BaseObjectLike.from_object(func)
    root = gast.parse(object_like.source_code)
    configuration = BaseTransformerConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    converter = MockGlobalsTransformer(root, transformer, configuration)
    # monkey-patch the transform_node method
    transformer.transform_node = converter.transform_node
    if apply_canonicalization:
        canonicalizer = BaseNameCanonicalizer(root, transformer, configuration)
        canonicalizer.transform()
        global_transformer = MockGlobalsTransformer(root, transformer, configuration)
        global_transformer.transform()
    else:
        converter.transform()
    return root, transformer


def test_nested_global_imports():
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_1 import (
        math_func,
    )

    root, transformer = transform_function(math_func)
    transformed = ast_to_source_code(root)

    assert "custom_sin(x) + MATH_OP.square(x) + PI * MATH_CONSTANT" in transformed

    global_str = [glob.assignment_str for glob in transformer.globals]
    assert (
        "MATH_OP = MathOperations()" in global_str
    )  # this would be "MATH_OP = Translated_MathOperations()" has we recursively transformed the RHS of the assignment
    assert "PI = 3.14159" in global_str
    assert "MATH_CONSTANT = 2.71828" in global_str


def test_inline_globals():
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_3 import (
        scale_data,
    )

    root, transformer = transform_function(scale_data)
    transformed = ast_to_source_code(root)

    assert (
        "data * PREPROCESS_CONSTANT * GLOB_2 + GLOB_1" in transformed
    )  # GLOB_2 should NOT get inlined and replaced with GLOB_1

    global_str = [glob.assignment_str for glob in transformer.globals]
    assert "PREPROCESS_CONSTANT = 1" in global_str
    assert "GLOB_1 = 10" in global_str
    assert "GLOB_2 = GLOB_1" in global_str


def test_ivy_globals():
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_4 import (
        ivy_function,
    )

    root, transformer = transform_function(ivy_function)
    transformed = ast_to_source_code(root)

    assert "ivy.sin(x) + pi" in transformed  # ivy.pi NOT inlined
    assert "res.dtype in valid_dtypes" in transformed  # ivy.valid_dtypes NOT inlined
    assert "lock = locks" in transformed  # ivy.locks NOT inlined
    global_str = [glob.assignment_str for glob in transformer.globals]
    assert "locks = {'backend_setter': threading.Lock()}" in global_str
    assert "pi = math.pi" in global_str
    assert "valid_dtypes = all_dtypes" in global_str


def test_torch_globals():
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_2 import (
        torch_func,
    )

    root, transformer = transform_function(torch_func)
    transformed = ast_to_source_code(root)

    global_str = [glob.assignment_str for glob in transformer.globals]
    assert "simplex.check(2) or real.check(1)" in transformed
    assert "simplex.check(10) or real.check(100)" in transformed

    assert (
        "torch.nn.functional.pad(torch.tensor([1, 2, 3]), (1, 1), mode='constant', value=0)"
        in transformed
    )

    assert (
        "torch.conv2d(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))" in transformed
    )
    assert (
        "torch.nn.functional.conv2d(torch.tensor([11, 22, 33]), torch.tensor([11, 22, 33]))"
        in transformed
    )

    assert (
        "torch.nn.functional.linear(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))"
        in transformed
    )

    global_str = [glob.assignment_str for glob in transformer.globals]

    assert "simplex = _Simplex()" in global_str
    assert "real = _Real()" in global_str
    assert len(global_str) == 2


def test_kornia_globals():
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_5 import (
        kornia_func,
    )

    root, transformer = transform_function(kornia_func)
    transformed = ast_to_source_code(root)

    assert "arr = tensor([1, 2, 3])" in transformed
    assert "arr = stack([arr, arr])" in transformed
    assert "param = Parameter(arr)" in transformed
    assert "mod = Module()" in transformed
    assert (
        "arr = pad(arr, (1, 1), mode='constant', value=0)" in transformed
    )  # pad(..) will not get directly captured. This is because it belongs from the `torch.nn.functional`
    # module which is flagged as an unsupported module.
    # If, however, we canonicalize pad(..) --> kornia.core.pad(..), now it'll get captured correctly.
    # this is because the globals transformer will correctly identify this as being an alias usage.
    root, transformer = transform_function(kornia_func, apply_canonicalization=True)
    transformed = ast_to_source_code(root)
    global_str = [glob.assignment_str for glob in transformer.globals]
    assert "pad = torch.nn.functional.pad" in global_str
    assert "stack = torch.stack" in global_str
    assert "tensor = torch.tensor" in global_str
    assert "Parameter = torch.nn.Parameter" in global_str
    assert "Module = torch.nn.Module" in global_str


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_hf_globals_fn_RHS_SR(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_6 import (
        hf_func_RHS_SR_global,
    )

    fn = hf_func_RHS_SR_global
    translated_fn = transpile(fn, source="torch", target=target)

    orig_inp = torch.tensor([1.0, 2.0, 3.0])
    translated_inp = _nest_torch_tensor_to_new_framework(orig_inp, target, False)

    orig_res = fn("gelu")(orig_inp)
    translated_res = translated_fn("gelu")(translated_inp)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_hf_globals_class_RHS_SR(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_6 import (
        HF_class_RHS_SR_global,
    )

    hf_cls = HF_class_RHS_SR_global
    translated_cls = transpile(hf_cls, source="torch", target=target)

    orig_inp = torch.tensor([1.0, 2.0, 3.0])
    translated_inp = _nest_torch_tensor_to_new_framework(orig_inp, target, False)

    orig_res = hf_cls().get_activation("gelu")(orig_inp)
    translated_res = translated_cls().get_activation("gelu")(translated_inp)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_hf_globals_fn_LHS_SR(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_6 import (
        hf_func_LHS_SR_global,
    )

    fn = hf_func_LHS_SR_global
    translated_fn = transpile(fn, source="torch", target=target)

    orig_inp = torch.tensor([1.0, 2.0, 3.0])
    translated_inp = _nest_torch_tensor_to_new_framework(orig_inp, target, False)

    orig_res = fn("gelu_10")(orig_inp)
    translated_res = translated_fn("gelu_10")(translated_inp)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_hf_globals_class_LHS_SR(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_6 import (
        HF_class_LHS_SR_global,
    )

    hf_cls = HF_class_LHS_SR_global
    translated_cls = transpile(hf_cls, source="torch", target=target)

    orig_inp = torch.tensor([1.0, 2.0, 3.0])
    translated_inp = _nest_torch_tensor_to_new_framework(orig_inp, target, False)

    orig_res = hf_cls().get_activation("gelu_10")(orig_inp)
    translated_res = translated_cls().get_activation("gelu_10")(translated_inp)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_cached_global_referenced_in_func(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_8 import (
        fn_with_cached_glob_used_as_obj,
    )

    fn = fn_with_cached_glob_used_as_obj
    translated_fn = transpile(fn, source="torch", target=target)

    orig_inp = torch.tensor([1.0, 2.0, 3.0])
    translated_inp = _nest_torch_tensor_to_new_framework(orig_inp, target, False)

    orig_res = fn("gelu_10")(orig_inp)
    translated_res = translated_fn("gelu_10")(translated_inp)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_cached_global_referenced_in_class(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_8 import (
        Class_with_cached_glob_used_as_obj,
    )

    cls = Class_with_cached_glob_used_as_obj
    translated_cls = transpile(cls, source="torch", target=target)

    orig_inp = torch.tensor([1.0, 2.0, 3.0])
    translated_inp = _nest_torch_tensor_to_new_framework(orig_inp, target, False)

    orig_res = cls().get_activation("gelu_10")(orig_inp)
    translated_res = translated_cls().get_activation("gelu_10")(translated_inp)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_cached_global_duplicated_in_func(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_8 import (
        fn_with_cached_glob_used_as_glob,
    )

    fn = fn_with_cached_glob_used_as_glob
    translated_fn = transpile(fn, source="torch", target=target)

    orig_inp = torch.tensor([1.0, 2.0, 3.0])
    translated_inp = _nest_torch_tensor_to_new_framework(orig_inp, target, False)

    orig_res = fn("gelu_10")(orig_inp)
    translated_res = translated_fn("gelu_10")(translated_inp)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_cached_global_duplicated_in_class(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_8 import (
        Class_with_cached_glob_used_as_glob,
    )

    cls = Class_with_cached_glob_used_as_glob
    translated_cls = transpile(cls, source="torch", target=target)

    orig_inp = torch.tensor([1.0, 2.0, 3.0])
    translated_inp = _nest_torch_tensor_to_new_framework(orig_inp, target, False)

    orig_res = cls().get_activation("gelu_10")(orig_inp)
    translated_res = translated_cls().get_activation("gelu_10")(translated_inp)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_cyclic_dependency_func(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_7 import (
        cyclic_dependency_func,
    )

    fn = cyclic_dependency_func
    translated_fn = transpile(fn, source="torch", target=target)

    orig_a, orig_b = fn()
    translated_a, translated_b = translated_fn()
    assert orig_a == translated_a
    assert orig_b == translated_b


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_cyclic_dependency_class(target):
    from ivy_tests.test_transpiler.transformations.globals_transformers.examples.func_7 import (
        Cyclic_dependency_class,
    )

    fn = Cyclic_dependency_class
    translated_fn = transpile(fn, source="torch", target=target)

    orig_a, orig_b = fn().echo()
    translated_a, translated_b = translated_fn().echo()
    assert orig_a == translated_a
    assert orig_b == translated_b


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_pad_partial_mixed_fn_globals(target):
    import ivy

    ivy.set_backend(target)

    orig_pad = ivy.pad
    translated_pad = transpile(orig_pad, source="ivy", target=target)

    x = ivy.array([[1, 2, 3], [4, 5, 6]])
    padding = ((1, 1), (2, 2))

    # Pad the array using the 'constant' mode (call backend implementation)
    orig_res = orig_pad(x, padding, mode="constant", constant_values=5)
    translated_res = translated_pad(x.data, padding, mode="constant", constant_values=5)
    _check_allclose(
        _nest_array_to_numpy(orig_res.data),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )
    # Create an input array
    x = ivy.array([[1, 2, 3], [4, 5, 6]])
    padding = ((1, 1), (2, 2))

    # Pad the array using the 'edge' mode (call compositional implementation)
    orig_res = orig_pad(x, padding, mode="edge")
    translated_res = translated_pad(x.data, padding, mode="edge")
    _check_allclose(
        _nest_array_to_numpy(orig_res.data),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_local_response_norm_partial_mixed_fn_globals(target):
    import ivy

    ivy.set_backend(target)

    orig_local_response_norm = ivy.local_response_norm
    translated_local_response_norm = transpile(
        orig_local_response_norm, source="ivy", target=target
    )

    x = ivy.random_normal(mean=0, std=1, shape=(8, 32, 32, 3))
    size = 11

    # local_response_norm the array using size % 2 != 0 (call backend implementation)
    orig_res = orig_local_response_norm(x, size)
    translated_res = translated_local_response_norm(x.data, size)
    _check_allclose(
        _nest_array_to_numpy(orig_res.data),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-3,
    )

    size = 10

    # local_response_norm the array using the size % 2 == 0 (call compositional implementation)
    orig_res = orig_local_response_norm(x, size)
    translated_res = translated_local_response_norm(x.data, size)
    _check_allclose(
        _nest_array_to_numpy(orig_res.data),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-3,
    )


@pytest.mark.parametrize(
    "target",
    ["tensorflow", "jax"],
)
def test_type_spec_globals(target):
    import ivy
    import numpy as np

    ivy.unset_backend()
    orig_asarray = ivy.asarray
    translated_asarray = transpile(orig_asarray, source="ivy", target=target)

    inp1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    inp2 = np.random.rand(2, 2)
    inp3 = 15.0

    # Input 1
    orig_res = orig_asarray(inp1)
    translated_res = translated_asarray(inp1)
    _check_allclose(
        _nest_array_to_numpy(orig_res.data),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )

    # Input 2
    orig_res = orig_asarray(inp2)
    translated_res = translated_asarray(inp2)
    _check_allclose(
        _nest_array_to_numpy(orig_res.data),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )

    # Input 3
    orig_res = orig_asarray(inp3)
    translated_res = translated_asarray(inp3)
    _check_allclose(
        _nest_array_to_numpy(orig_res.data),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


if __name__ == "__main__":
    pytest.main([__file__])
