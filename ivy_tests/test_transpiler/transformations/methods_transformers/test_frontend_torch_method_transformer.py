import pytest
import gast
import torch
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import ConfigurationsContainer
from ivy.transpiler.utils.ast_utils import ast_to_source_code
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformers.method_transformer.frontend_torch_method_transformer import (
    FrontendTorchMethodToFunctionConverter,
)
from ivy.transpiler import transpile
from ivy_tests.test_transpiler.kornia.helpers import (
    _nest_torch_tensor_to_new_framework,
    _check_allclose,
    _nest_array_to_numpy,
)


def transform_function(func):
    # Set up the configurations container
    container = ConfigurationsContainer()
    container.load_configurations(source="torch_frontend", target="ivy")

    # Create BaseObjectLike from the function
    object_like = BaseObjectLike.from_object(func)
    root = gast.parse(object_like.source_code)

    # Instantiate the transformer and transform the object
    configuration = BaseTransformerConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    converter = FrontendTorchMethodToFunctionConverter(root, transformer, configuration)
    converter.transform()

    # Return the transformed source code
    return ast_to_source_code(root).strip()


# Test cases
def test_simple_method_call():
    def func(x):
        return x.flatten()

    transformed = transform_function(func)
    assert "torch.Tensor.flatten(x)" in transformed


def test_method_call_with_arguments():
    def func(x):
        return x.reshape(1, 2, 3)

    transformed = transform_function(func)
    assert "torch.Tensor.reshape(x, 1, 2, 3)" in transformed


def test_chained_method_calls():
    def func(x):
        return x.flatten().sum()

    transformed = transform_function(func)
    assert "torch.Tensor.sum(torch.Tensor.flatten(x))" in transformed


def test_nested_method_calls():
    def func(x, y, z):
        return x.to(y.device).type(z.dtype)

    transformed = transform_function(func)
    assert "torch.Tensor.type(torch.Tensor.to(x, y.device), z.dtype)" in transformed


def test_property_access_not_transformed():
    def func():
        x = x.dtype
        xx = x.data.dtype
        xxx = x.ivy_array.dtype

        y = x.device
        yy = x.data.device
        yyy = x.ivy_array.device

        z = x.ivy_array
        zzz = x.ivy_array.ivy_array

    transformed = transform_function(func)
    assert "x = x.dtype" in transformed
    assert "xx = torch.Tensor.data(x).dtype" in transformed
    assert "xxx = x.ivy_array.dtype" in transformed

    assert "y = x.device" in transformed
    assert "yy = torch.Tensor.data(x).device" in transformed
    assert "yyy = x.ivy_array.device" in transformed

    assert "z = x.ivy_array" in transformed
    assert "zzz = x.ivy_array.ivy_array" in transformed


def test_property_access_transformed():
    class TestClass:
        @property
        def custom_property(self):
            pass

        def custom_method(self):
            x = self.weight.T
            y = self.custom_property
            z = self.bias.ndim

            shp = x.shape
            shp = x.data.shape
            shp = x.ivy_array.shape

    transformed = transform_function(TestClass)
    assert "x = torch.Tensor.T(self.weight)" in transformed
    assert "y = self.custom_property" in transformed
    assert "z = torch.Tensor.ndim(self.bias)" in transformed

    assert "shp = torch.Tensor.shape(x)" in transformed
    assert "shp = torch.Tensor.shape(torch.Tensor.data(x))" in transformed
    assert "shp = x.ivy_array.shape" in transformed


def test_builtin_method_not_transformed():
    def func(x):
        x.append(5)
        x.insert(5)
        x.pop(5)
        x.remove(5)

    transformed = transform_function(func)
    assert "x.append(5)" in transformed
    assert "x.insert(5)" in transformed
    assert "x.pop(5)" in transformed
    assert "x.remove(5)" in transformed


def test_super_call_not_transformed():
    class TestClass:

        def __init__(self):
            super().__init__()

        def __getattr__(self):
            super(self, TestClass).__getattr__()
            super().__getattr__().__setattr__()
            super().method().__setattr__()
            super(self, TestClass).__getattr__().flatten()
            super().__getattr__().T

        def method(self):
            super(self, TestClass).flatten()
            super(self, TestClass).method()
            super().method().dummy_method()
            super(self, TestClass).method().flatten()
            super().method().T

    transformed = transform_function(TestClass)
    assert "super(self, TestClass).__getattr__()" in transformed
    assert "super().__getattr__().__setattr__()" in transformed
    assert "super().method().__setattr__()" in transformed
    assert "torch.Tensor.flatten(super(self, TestClass).__getattr__())" in transformed
    assert "torch.Tensor.T(super().__getattr__()" in transformed

    assert "super(self, TestClass).flatten()" in transformed
    assert "super(self, TestClass).method()" in transformed
    assert "super().method().dummy_method()" in transformed
    assert "torch.Tensor.flatten(super(self, TestClass).method())" in transformed
    assert "torch.Tensor.T(super().method())" in transformed
    assert "super().method()" in transformed


def test_user_defined_method_not_transformed():
    class TestClass:
        def custom_method(self):
            return self.custom_method()

    transformed = transform_function(TestClass)
    assert "self.custom_method()" in transformed


def test_torch_function_not_transformed():
    def func():
        x = torch.tensor([1, 2, 3])
        y = torch.cos(x)
        z = torch.nn.Conv2d(3, 64, 3)

    transformed = transform_function(func)
    assert "torch.tensor([1, 2, 3])" in transformed
    assert "torch.cos(x)" in transformed
    assert "torch.nn.Conv2d(3, 64, 3)" in transformed

def test_conflicting_variable_names():
    def new_f():
       fx: torch.Tensor = torch.tensor(1)
       nn = torch.tensor([1,2.])
       a = fx.unsqueeze()
       b = nn.mean()

    transformed = transform_function(new_f)
    assert "a = torch.Tensor.unsqueeze(fx)" in transformed
    assert "b = torch.Tensor.mean(nn)" in transformed

def test_complex_scenario():
    def func(x, y, z):
        return x.flatten().reshape(y.size()).to(z.device().type("cuda"))[..., 10]

    transformed = transform_function(func)
    print(transformed)
    expected = "torch.Tensor.to(torch.Tensor.reshape(torch.Tensor.flatten(x), torch.Tensor.size(y)), torch.Tensor.type(torch.Tensor.device(z), 'cuda'))[..., 10]"
    print(expected)
    assert expected in transformed


def test_complex_scenario_2():
    def func(x_mask):
        patch_index = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1])
                ),
                dim=-1,
            )[None, None, :, :, :]
            .expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
            .flatten(1, 3)
        )

    transformed = transform_function(func)
    expected_transformations = [
        "torch.Tensor.flatten(",
        "torch.Tensor.expand(",
        "torch.stack(",
        "torch.meshgrid(",
        "torch.arange(",
        "torch.Tensor.shape(x_mask)[",
        "dim=-1)[None, None, :, :, :]",
    ]

    for expected in expected_transformations:
        assert expected in transformed, f"Expected '{expected}' in transformed code"

    # Check for proper nesting of function calls
    assert "torch.Tensor.flatten(torch.Tensor.expand(" in transformed
    assert "torch.stack(torch.meshgrid(" in transformed


def test_indexing_with_method_call():
    def func(x, y):
        return x[y.argmax()].flatten()

    transformed = transform_function(func)
    assert "torch.Tensor.flatten(x[torch.Tensor.argmax(y)])" in transformed


def test_slicing_with_method_call():
    def func(x, y):
        return x[y.nonzero().squeeze()].mean()

    transformed = transform_function(func)
    assert (
        "torch.Tensor.mean(x[torch.Tensor.squeeze(torch.Tensor.nonzero(y))])"
        in transformed
    )


def test_complex_indexing_and_method_chaining():
    def func(x, y, z):
        return x[y.argmax() :].reshape(-1).to(z.device())

    transformed = transform_function(func)
    assert (
        "torch.Tensor.to(torch.Tensor.reshape(x[torch.Tensor.argmax(y):], -1), torch.Tensor.device(z))"
        in transformed
    )


def test_multidimensional_indexing_with_methods():
    def func(x, y):
        return x[y.sort()[0], :].transpose(0, 1).contiguous()

    transformed = transform_function(func)
    expected = "torch.Tensor.contiguous(torch.Tensor.transpose(x[torch.Tensor.sort(y)[0], :], 0, 1))"
    assert expected in transformed


def test_indexing_with_multiple_tensors():
    def func(x, y, z):
        return x[y.bool(), z.long()].sum(dim=0)

    transformed = transform_function(func)
    expected = "torch.Tensor.sum(x[torch.Tensor.bool(y), torch.Tensor.long(z)], dim=0)"
    assert expected in transformed


def test_nested_indexing_and_method_calls():
    def func(x, y, z):
        return x[y.argmax(dim=1)][:, z.topk(k=5)].norm()

    transformed = transform_function(func)
    expected = "torch.Tensor.norm(x[torch.Tensor.argmax(y, dim=1)][:, torch.Tensor.topk(z, k=5)])"
    assert expected in transformed


def test_multiple_statements():
    def func():
        x = torch.tensor([1, 2, 3])
        y = x.flatten()
        z = y.sum()
        return z

    transformed = transform_function(func)
    assert "x = torch.tensor([1, 2, 3])" in transformed
    assert "y = torch.Tensor.flatten(x)" in transformed
    assert "z = torch.Tensor.sum(y)" in transformed


def test_method_call_in_function_args():
    def func(x, y):
        return torch.cat([x.flatten(), y.reshape(-1)])

    transformed = transform_function(func)
    assert (
        "torch.cat([torch.Tensor.flatten(x), torch.Tensor.reshape(y, -1)])"
        in transformed
    )


def test_method_call_in_list_comprehension():
    def func(tensor_list):
        return [x.item() for x in tensor_list]

    transformed = transform_function(func)
    assert "[torch.Tensor.item(x) for x in tensor_list]" in transformed


def test_method_call_in_dict_comprehension():
    def func(tensor_list):
        return {i: x.sum() for i, x in enumerate(tensor_list)}

    transformed = transform_function(func)
    assert "{i: torch.Tensor.sum(x) for i, x in enumerate(tensor_list)}" in transformed


def test_method_call_in_lambda():
    def func():
        return lambda x: x.flatten()

    transformed = transform_function(func)
    assert "lambda x: torch.Tensor.flatten(x)" in transformed


W = torch.tensor([1, 2, 3])


def test_method_call_in_global_var():
    def func():
        return W.to("cpu")

    transformed = transform_function(func)
    assert "torch.Tensor.to(W, 'cpu')" in transformed


def test_method_call_already_transformed():
    def func(self, memo):
        a = torch.Tensor.add(memo, id(self))
        b = torch.Tensor.flatten(self).arccos()

    transformed = transform_function(func)
    print(transformed)
    assert "torch.Tensor.add(memo, id(self))" in transformed
    assert "torch.Tensor.arccos(torch.Tensor.flatten(self))" in transformed


def test_method_call_with_conflicting_builtin_name():
    def func(tensor):
        center = tensor
        add = center.expand(
            tensor.shape[0], -1
        )  # eg: "center" conflicts builtin method <str.center>
        return add.sum(axis=-1)  # eg: "add" conflicts builtin method <set.add>

    transformed = transform_function(func)
    assert (
        "torch.Tensor.expand(center, torch.Tensor.shape(tensor)[0], -1)" in transformed
    )
    assert "torch.Tensor.sum(add, axis=-1)" in transformed


def test_numel_method_call():
    class DummyCls:
        def __init__(self, a, b):
            self._data = a
            self._mode = b

        @property
        def shape(self):
            return self._data.shape

        def numel(self):
            return self._data.numel()

        def method(self, x):
            shp = self.numel()
            shp = self.shape.numel()
            shp = self.data.shape.numel()

            shp = x.numel()
            shp = x.shape.numel()
            shp = x.data.shape.numel()

    import ivy.transpiler.transformations.transformer_globals as glob

    glob.CONFLICTING_METHODS = set()

    transformed = transform_function(DummyCls)
    assert "self.numel()" in transformed
    assert "self.shape.numel()" in transformed
    assert "torch.Tensor.shape(self.data).numel()" in transformed
    assert "torch.Tensor.numel(x)" in transformed
    assert "torch.Tensor.shape(x).numel()" in transformed
    assert "torch.Tensor.shape(torch.Tensor.data(x)).numel()" in transformed

    assert "shape" in glob.CONFLICTING_METHODS
    assert "numel" in glob.CONFLICTING_METHODS


def test_method_call_with_conflicting_cls_method_name():
    class DummyCls:
        def __init__(self, a, b):
            self._data = a
            self._mode = b

        @property
        def shape(self):
            return self.data.shape

        @property
        def data(self):
            return self._data

        def index_put_(self, indices, values=False, inplace=False):
            _data = self._data
            _data = self._data.clone()  # clone refers to torch.Tensor.data
            self._data.index_put_(
                indices, values.data
            )  # index_put refers to torch.Tensor.index_put

            obj = self.clone()  # clone refers to DummyCls.clone
            self.index_put_(indices, values)  # index_put refers to DummyCls.index_put

            return obj

        def clone(self):
            pass

    import ivy.transpiler.transformations.transformer_globals as glob

    glob.CONFLICTING_METHODS = set()

    transformed = transform_function(DummyCls)

    assert "torch.Tensor.clone(self._data)" in transformed
    assert (
        "torch.Tensor.index_put_(self._data, indices, torch.Tensor.data(values))"
        in transformed
    )

    assert "obj = self.clone()" in transformed
    assert "self.index_put_(indices, values)" in transformed

    assert "data" in glob.CONFLICTING_METHODS
    assert "clone" in glob.CONFLICTING_METHODS
    assert "index_put_" in glob.CONFLICTING_METHODS


def test_method_call_with_conflicting_property_name():
    class DummyCls:
        def __init__(self, a, b):
            self._data = a
            self._mode = b

        @property
        def shape(self):
            return self.data.shape

        @property
        def data(self):
            return self._data

        @property
        def T(self):
            return self._data.T

        def compute_transpose(self, x):
            _t = self.T  # T refers to DummyCls.T
            _t = self.data.T  # clone refers to torch.Tensor.T

            _data = self.data  # data refers to DummyCls.data
            _data = self.data.data  # data refers to torch.Tensor.data

            shp = self.shape  # shape refers to DummyCls.shape
            shp = self.data.shape  # shape refers to torch.Tensor.shape

    transformed = transform_function(DummyCls)
    assert "_t = self.T" in transformed
    assert "_data = self.data" in transformed
    assert "shp = self.shape" in transformed

    assert "torch.Tensor.T(self.data)" in transformed
    assert "torch.Tensor.data(self.data)" in transformed
    assert "torch.Tensor.shape(self.data)" in transformed

    import ivy.transpiler.transformations.transformer_globals as glob

    assert "T" in glob.CONFLICTING_METHODS
    assert "data" in glob.CONFLICTING_METHODS
    assert "shape" in glob.CONFLICTING_METHODS


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_torch_numpy_call(target):
    import torch

    def numpy_func(tensor):
        return tensor.cpu().detach().numpy()

    translated_fn = transpile(numpy_func, target=target)
    data = torch.rand(10, 10)
    translated_data = _nest_torch_tensor_to_new_framework(data, target, True)
    orig_res = numpy_func(data)
    translated_res = translated_fn(translated_data)
    _check_allclose(
        orig_res,
        translated_res,
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_torch_dunders(target):
    import torch

    def dunder_func(x):
        w = x / 1.5
        return w

    data = torch.tensor(10)
    translated_data = _nest_torch_tensor_to_new_framework(data, target, False)
    try:
        # this will throw an error as tf.Tensor does not support '/' with incompatible dtypes
        dunder_func(translated_data)
    except Exception as e:
        print(e)
        translated_fn = transpile(dunder_func, target=target)
        orig_res = dunder_func(data)
        # this will pass because we now monkey patch tf.Tensor.__truediv__
        translated_res = translated_fn(translated_data)
        _check_allclose(
            _nest_array_to_numpy(orig_res),
            _nest_array_to_numpy(translated_res),
            tolerance=1e-2,
        )


if __name__ == "__main__":
    pytest.main([__file__])
