import pytest
import gast
import ivy
from ivy.utils.backend import current_backend
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import ConfigurationsContainer
from ivy.transpiler.utils.ast_utils import ast_to_source_code
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformers.method_transformer.ivy_method_transformer import (
    IvyMethodToFunctionConverter,
)


def transform_function(func, target="tensorflow"):
    # Set up the configurations container
    container = ConfigurationsContainer()
    container.load_configurations(source="ivy", target=target)

    # Create BaseObjectLike from the function
    object_like = BaseObjectLike.from_object(func, target)
    root = gast.parse(object_like.source_code)

    # Instantiate the transformer and transform the object
    configuration = BaseTransformerConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    converter = IvyMethodToFunctionConverter(root, transformer, configuration)
    converter.transform()

    # Return the transformed source code
    return ast_to_source_code(root).strip()


# Test cases
@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_simple_method_call(target):
    def func(x):
        return x.flatten()

    transformed = transform_function(func, target)
    assert "ivy.Array.flatten(x)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_with_arguments(target):
    def func(x):
        return x.reshape((1, 2, 3))

    transformed = transform_function(func, target)
    assert "ivy.Array.reshape(x, (1, 2, 3))" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_chained_method_calls(target):
    def func(x):
        return x.flatten().sum()

    transformed = transform_function(func, target)
    assert "ivy.Array.sum(ivy.Array.flatten(x))" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_nested_method_calls(target):
    def func(x, y, z):
        return x.to_device(y.device).astype(z.dtype)

    transformed = transform_function(func, target)
    assert "ivy.Array.astype(ivy.Array.to_device(x, y.device), z.dtype)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_property_access_not_transformed(target):
    def func():
        x = x.dtype
        xx = x.data.dtype
        xxx = x._data.dtype

        y = x.device
        yy = x.data.device
        yyy = x._data.device

        shp = x.shape
        shp = x.data.shape
        shp = x._data.shape

        z = x.data
        zzz = x._data.data

        w = x.strides
        ww = x.data.strides
        www = x._data.strides

    transformed = transform_function(func, target)
    print(transformed)
    assert "x = x.dtype" in transformed
    assert "xx = x.data.dtype" in transformed
    assert "xxx = x._data.dtype" in transformed

    assert "y = x.device" in transformed
    assert "yy = x.data.device" in transformed
    assert "yyy = x._data.device" in transformed

    assert "shp = x.shape" in transformed
    assert "shp = x.data.shape" in transformed
    assert "shp = x._data.shape" in transformed

    assert "z = x.data" in transformed
    assert "zzz = x._data.data" in transformed

    assert "w = x.strides" in transformed
    assert "ww = x.data.strides" in transformed
    assert "www = x._data.strides" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "jax",
    ],
)
def test_property_access_not_transformed_jax(target):
    def func():
        x = x.size
        x = x.data.size
        x = x._data.size

        xx = x.ndim
        xx = x.data.ndim
        xx = x._data.ndim

        xxx = x.itemsize
        xxx = x.data.itemsize
        xxx = x._data.itemsize

    transformed = transform_function(func, target)
    print(transformed)
    assert "x = x.size" in transformed
    assert "x = x.data.size" in transformed
    assert "x = x._data.size" in transformed

    assert "xx = x.ndim" in transformed
    assert "xx = x.data.ndim" in transformed
    assert "xx = x._data.ndim" in transformed

    assert "xxx = x.itemsize" in transformed
    assert "xxx = x.data.itemsize" in transformed
    assert "xxx = x._data.itemsize" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "jax",
    ],
)
def test_jax_array_at_calls(target):
    def func(y):
        x = x.at[0].min()
        x = x.at[y].max()
        x = x.at[:y, 0].set(0)

    transformed = transform_function(func, target)
    print(transformed)
    assert "x = x.at[0].min()" in transformed
    assert "x = x.at[y].max()" in transformed
    assert "x = x.at[:y, 0].set(0)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_property_access_transformed(target):
    class TestClass:
        @property
        def custom_property(self):
            pass

        def custom_method(self):
            x = self.weight.mT
            y = self.custom_property
            z = self.bias.real

    transformed = transform_function(TestClass)
    assert "x = ivy.Array.mT(self.weight)" in transformed
    assert "y = self.custom_property" in transformed
    assert "z = ivy.Array.real(self.bias)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_builtin_method_not_transformed(target):
    def func(x):
        x.append(5)
        x.insert(5)
        x.pop(5)
        x.remove(5)

    transformed = transform_function(func, target)
    assert "x.append(5)" in transformed
    assert "x.insert(5)" in transformed
    assert "x.pop(5)" in transformed
    assert "x.remove(5)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_super_call_not_transformed(target):
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

    transformed = transform_function(TestClass, target)
    assert "super(self, TestClass).__getattr__()" in transformed
    assert "super().__getattr__().__setattr__()" in transformed
    assert "super().method().__setattr__()" in transformed
    assert "ivy.Array.flatten(super(self, TestClass).__getattr__())" in transformed
    assert "ivy.Array.T(super().__getattr__()" in transformed

    assert "super(self, TestClass).flatten()" in transformed
    assert "super(self, TestClass).method()" in transformed
    assert "super().method().dummy_method()" in transformed
    assert "ivy.Array.flatten(super(self, TestClass).method())" in transformed
    assert "ivy.Array.T(super().method())" in transformed
    assert "super().method()" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_current_backend_not_transformed(target):
    def func(x):
        current_backend(x).size(x)
        current_backend(x).add(x)
        current_backend(x).prod(x)

    transformed = transform_function(func, target)
    assert "current_backend(x).size(x)" in transformed
    assert "current_backend(x).add(x)" in transformed
    assert "current_backend(x).prod(x)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_user_defined_method_not_transformed(target):
    class TestClass:
        def custom_method(self):
            return self.custom_method()

    transformed = transform_function(TestClass.custom_method)
    assert "self.custom_method()" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_ivy_function_not_transformed(target):
    def func():
        x = ivy.array([1, 2, 3])
        y = ivy.cos(x)
        z = ivy.nested_map(lambda w: w, x)

    transformed = transform_function(func, target)
    assert "ivy.array([1, 2, 3])" in transformed
    assert "ivy.cos(x)" in transformed
    assert "ivy.nested_map(lambda w: w, x)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_complex_scenario(target):
    def func(x, y, z):
        return x.flatten().reshape(y.size).to_device(z.device)

    transformed = transform_function(func, target)
    if target == "jax":
        expected = "ivy.Array.to_device(ivy.Array.reshape(ivy.Array.flatten(x), y.size), z.device)"  # y.size is not transformed
    else:
        expected = "ivy.Array.to_device(ivy.Array.reshape(ivy.Array.flatten(x), ivy.Array.size(y)), z.device)"
    assert expected in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_complex_scenario_2(target):
    def func(x_mask):
        patch_index = (
            ivy.stack(
                ivy.meshgrid(
                    ivy.arange(x_mask.shape[-2]), ivy.arange(x_mask.shape[-1])
                ),
                axis=-1,
            )[None, None, :, :, :]
            .expand((x_mask.shape[0], x_mask.shape[1], -1, -1, -1))
            .flatten(start_dim=1, end_dim=3)
        )
        return patch_index

    transformed = transform_function(func, target)
    print(transformed)
    expected_transformations = [
        "ivy.Array.flatten(",
        "ivy.Array.expand(",
        "ivy.stack(",
        "ivy.meshgrid(",
        "ivy.arange(",
        "x_mask.shape[",
        "axis=-1)[None, None, :, :, :]",
    ]

    for expected in expected_transformations:
        assert expected in transformed, f"Expected '{expected}' in transformed code"

    # Check for proper nesting of function calls
    assert "ivy.Array.flatten(ivy.Array.expand(" in transformed
    assert "ivy.stack(ivy.meshgrid(" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_indexing_with_method_call(target):
    def func(x, y):
        return x[y.argmax()].flatten()

    transformed = transform_function(func, target)
    assert "ivy.Array.flatten(x[ivy.Array.argmax(y)])" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_slicing_with_method_call(target):
    def func(x, y):
        return x[y.nonzero().squeeze()].mean()

    transformed = transform_function(func, target)
    assert "ivy.Array.mean(x[ivy.Array.squeeze(ivy.Array.nonzero(y))])" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_complex_indexing_and_method_chaining(target):
    def func(x, y, z):
        return x[y.argmax() :].reshape((-1,)).to_device(z.device)

    transformed = transform_function(func, target)
    assert (
        "ivy.Array.to_device(ivy.Array.reshape(x[ivy.Array.argmax(y):], (-1,)), z.device)"
        in transformed
    )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_multidimensional_indexing_with_methods(target):
    def func(x, y):
        return x[y.sort()[0], :].T.asarray()

    transformed = transform_function(func, target)
    expected = "ivy.Array.asarray(ivy.Array.T(x[ivy.Array.sort(y)[0], :]))"
    assert expected in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_indexing_with_multiple_arrays(target):
    def func(x, y, z):
        return x[y.astype("bool"), z.astype("int64")].sum(axis=0)

    transformed = transform_function(func, target)
    expected = "ivy.Array.sum(x[ivy.Array.astype(y, 'bool'), ivy.Array.astype(z, 'int64')], axis=0)"
    assert expected in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_nested_indexing_and_method_calls(target):
    def func(x, y, z):
        return x[y.argmax(axis=1)][:, z.view(-1)].vector_norm()

    transformed = transform_function(func, target)
    expected = "ivy.Array.vector_norm(x[ivy.Array.argmax(y, axis=1)][:, ivy.Array.view(z, -1)])"
    assert expected in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_multiple_statements(target):
    def func():
        x = ivy.asarray([1, 2, 3])
        y = x.flatten()
        z = y.sum()
        return z

    transformed = transform_function(func, target)
    assert "x = ivy.asarray([1, 2, 3])" in transformed
    assert "y = ivy.Array.flatten(x)" in transformed
    assert "z = ivy.Array.sum(y)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_in_function_args(target):
    def func(x, y):
        return ivy.concat([x.flatten(), y.reshape(-1)])

    transformed = transform_function(func, target)
    assert "ivy.concat([ivy.Array.flatten(x), ivy.Array.reshape(y, -1)])" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_in_list_comprehension(target):
    def func(tensor_list):
        return [x.to_numpy() for x in tensor_list]

    transformed = transform_function(func, target)
    assert "[ivy.Array.to_numpy(x) for x in tensor_list]" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_in_dict_comprehension(target):
    def func(tensor_list):
        return {i: x.sum() for i, x in enumerate(tensor_list)}

    transformed = transform_function(func, target)
    assert "{i: ivy.Array.sum(x) for i, x in enumerate(tensor_list)}" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_in_lambda(target):
    def func():
        return lambda x: x.flatten()

    transformed = transform_function(func, target)
    assert "lambda x: ivy.Array.flatten(x)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_not_to_be_transformed(target):
    def func(x):
        return x.current_backend()

    transformed = transform_function(func, target)
    assert "x.current_backend()" in transformed


W = ivy.array([1, 2, 3])


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_in_global_var(target):
    def func():
        return W.to_device("cpu")

    transformed = transform_function(func, target)
    assert "ivy.Array.to_device(W, 'cpu')" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_already_transformed(target):
    def func(self, memo):
        a = ivy.Array.add(memo, id(self))
        b = ivy.Array.flatten(self).cos()

    transformed = transform_function(func, target)
    print(transformed)
    assert "ivy.Array.add(memo, id(self))" in transformed
    assert "ivy.Array.cos(ivy.Array.flatten(self))" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_with_conflicting_builtin_name(target):
    def func(tensor):
        center = tensor
        add = center.expand(
            tensor.shape[0], -1
        )  # eg: "center" conflicts builtin method <str.center>
        return add.sum(axis=-1)  # eg: "add" conflicts builtin method <set.add>

    transformed = transform_function(func, target)
    assert "ivy.Array.expand(center, tensor.shape[0], -1)" in transformed
    assert "ivy.Array.sum(add, axis=-1)" in transformed


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_with_conflicting_cls_method_name(target):
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

        def multiply(self, indices, values=False, inplace=False):
            _data = self._data
            _data = self._data.mean()  # ones refers to ivy.Array.ones
            _data = self.data.mean()  # ones refers to ivy.Array.ones

            self.data.multiply(values.data)  # multiply refers to ivy.Array.multiply

            obj = self.mean(_data)  # ones refers to DummyCls.ones
            self.multiply(indices, values)  # multiply refers to DummyCls.multiply

            return obj

        def mean(self):
            pass

    import ivy.transpiler.transformations.transformer_globals as glob

    glob.CONFLICTING_METHODS = set()
    transformed = transform_function(DummyCls, target)

    assert "ivy.Array.mean(self.data)" in transformed
    assert "ivy.Array.mean(self._data)" in transformed
    assert "ivy.Array.multiply(self.data, values.data)" in transformed

    assert "obj = self.mean(_data)" in transformed
    assert "self.multiply(indices, values)" in transformed

    assert "mean" in glob.CONFLICTING_METHODS
    assert "multiply" in glob.CONFLICTING_METHODS
    len(glob.CONFLICTING_METHODS) == 2


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_method_call_with_conflicting_property_name(target):
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

        def compute_transpose(
            self,
        ):
            _t = self.T  # T refers to DummyCls.T
            _t = self.data.T  # clone refers to ivy.Array.T

            _data = self.data  # data refers to DummyCls.data
            _data = self.data.data  # shape refers to ivy.Array.data.data

            shp = self.shape  # shape refers to DummyCls.shape
            shp = self.data.shape  # shape refers to ivy.Array.shape

    import ivy.transpiler.transformations.transformer_globals as glob

    glob.CONFLICTING_METHODS = set()

    transformed = transform_function(DummyCls, target)
    assert "_t = self.T" in transformed
    assert "_data = self.data" in transformed
    assert "shp = self.shape" in transformed

    assert "ivy.Array.T(self.data)" in transformed
    assert "self.data.data" in transformed
    assert "self.data.shape" in transformed

    assert "T" in glob.CONFLICTING_METHODS
    len(glob.CONFLICTING_METHODS) == 1


if __name__ == "__main__":
    pytest.main([__file__])
