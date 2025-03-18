import pytest
import gast
import os
import ivy
from ivy.transpiler.transformations.configurations.ivy_postprocessing_transformer_config import (
    IvyCodePostProcessorConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.postprocessing_transformer.ivy_to_tf_postprocessing_transformer import (
    IvyToTFCodePostProcessor,
)
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import ConfigurationsContainer
from ivy.transpiler.utils.ast_utils import ast_to_source_code


def transform_function(func, apply_fn: callable = None):
    ivy.set_backend("tensorflow")
    # Set up the configurations container
    container = ConfigurationsContainer()
    container.load_configurations(source="ivy", target="tensorflow")

    # Create BaseObjectLike from the function
    object_like = BaseObjectLike.from_object(func, root_obj=func)
    root = gast.parse(object_like.source_code)

    if apply_fn:
        object_like = apply_fn(object_like)
    # Instantiate the transformer and transform the object
    configuration = IvyCodePostProcessorConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    converter = IvyToTFCodePostProcessor(root, transformer, configuration)
    converter.transform()

    # Return the transformed source code
    return ast_to_source_code(root).strip()


# Test cases
def test_hf_class_mapping():
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import BaseModelOutput

    class Foo(PreTrainedModel):
        def forward(self, x):
            return BaseModelOutput

    transformed = transform_function(Foo)

    assert "class Foo(TFPreTrainedModel)" in transformed
    assert "def call(self, x):" in transformed
    assert "return TFBaseModelOutput" in transformed


def test_ivy_Array_class_attributes():

    def foo(x):
        x.data
        x._data

    # not an ivy api so attributes should not be deleted
    transformed = transform_function(foo)
    assert "x.data" in transformed
    assert "x._data" in transformed

    class Foo:
        def __init__(self, x):
            self.x = x.data
            self.y = x._data

    # not an ivy api so attributes should not be deleted
    transformed = transform_function(Foo)
    assert "self.x = x.data" in transformed
    assert "self.y = x._data" in transformed

    # part of ivy api so attributes should be deleted
    transformed = transform_function(ivy.Array.searchsorted)
    assert (
        "return ivy.searchsorted(self, v, side=side, sorter=sorter, ret_dtype=ret_dtype, out=out)"
        in transformed
    )


def test_rename_arr():
    def foo(arr):
        x = arr

    def apply_fn(object_like):
        object_like.is_translated_api = True
        return object_like

    transformed = transform_function(foo, apply_fn=apply_fn)
    assert "def foo(tensor)" in transformed
    assert "x = tensor" in transformed


def test_visit_Import():
    def foo():
        import ivy

    transformed = transform_function(foo)
    assert "import ivy" not in transformed


def test_visit_ImportFrom():
    def foo():
        from ivy import asarray

    transformed = transform_function(foo)
    assert "from ivy import asarray" not in transformed


def test_visit_ClassDef():
    class Foo(ivy.Module):
        pass

    transformed = transform_function(Foo)
    assert "class Foo(tensorflow_keras_Model):" in transformed


def test_visit_FunctionDef():
    class Foo(ivy.Module):
        def forward(x):
            return x

    transformed = transform_function(Foo)
    assert "def forward(x):" in transformed


def test_visit_arguments():
    def foo(x: ivy.Array, y: int = 0):
        pass

    transformed = transform_function(foo)
    assert "x: tensorflow.Tensor, y: int=0" in transformed


def test_visit_Name():
    def foo():
        x = ivy.float32

    transformed = transform_function(foo)
    assert "x = tf.float32" in transformed


def test_visit_Attribute():
    def foo():
        x = ivy.Array(5)
        y = ivy.exceptions.IvyException
        z = ivy.inplace_update(x).data

    transformed = transform_function(foo)
    assert "x = tensorflow.convert_to_tensor(5)" in transformed
    assert "y = Exception" in transformed
    assert "z = ivy.inplace_update(x)" in transformed


def test_visit_Attribute2():
    class Foo(ivy.Module):
        def __init__(
            self,
        ):
            self.x = ivy.Variable(5)
            self.weight = ivy.Variable(5)

    transformed = transform_function(Foo)
    assert "self.x = tensorflow.keras.Variable(5)" in transformed
    assert "self.weight = tensorflow.keras.Variable(5)" in transformed


def test_visit_With():
    def foo():
        with ivy.ArrayMode():
            pass

    transformed = transform_function(foo)
    assert "with ivy.ArrayMode():" not in transformed


def test_visit_Call():
    def foo():
        x = ivy.default_float_dtype(as_native=True)

    transformed = transform_function(foo)
    assert "x = tf.float32" in transformed


def test_convert_list_comps():
    def foo():
        x = [i for i in range(10)]

    transformed = transform_function(foo)
    assert "ag__result_list_0 = []" in transformed
    assert "for i in range(10):" in transformed
    assert "ag__result_list_0.append(res)" in transformed


def test_handle_ivy_array():
    def foo():
        x = ivy.Array([1, 2, 3])

    transformed = transform_function(foo)
    assert "x = tensorflow.convert_to_tensor([1, 2, 3])" in transformed


def test_handle_isinstance():
    def foo(x):
        return isinstance(x, (ivy.Array, ivy.Array))

    transformed = transform_function(foo)
    assert "isinstance(x, (tensorflow.Tensor, tensorflow.Variable))" in transformed


def test_isinstance_ivy_array():

    class ivy_Vector3:
        pass

    def init(rotation, translation):
        if not isinstance(translation, (ivy_Vector3, ivy.Array, ivy.Array)):
            raise TypeError(f"translation type is {type(translation)}")

    transformed = transform_function(init)

    assert (
        "if not isinstance(translation, (ivy_Vector3, tensorflow.Tensor, tensorflow.Variable)):"
        in transformed
    )


def test_maybe_add_unpack_inputs_decorator():
    from transformers import PreTrainedModel

    class Foo(PreTrainedModel):
        def forward(self, x):
            pass

    transformed = transform_function(Foo)
    assert "@unpack_inputs" in transformed
    assert "def call(self, x):" in transformed


def test_maybe_replace_with_native_array_calls():
    import tensorflow

    def foo():
        x = tensorflow.Tensor([1, 2, 3])

    transformed = transform_function(foo)
    assert "x = tensorflow.convert_to_tensor([1, 2, 3])" in transformed


def test_conv_transpose_optimiz_parameters():
    import tensorflow as tf

    tensorflow_empty_frnt = lambda shape, device, dtype: tf.experimental.numpy.empty(
        shape, device=device, dtype=dtype
    )

    class tensorflow__ConvNd(tf.keras.Model):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            transposed,
            groups,
            device=None,
            dtype=None,
            **kwargs,
        ):

            factory_kwargs = {"device": device, "dtype": dtype}
            if transposed:
                self.weight1 = tf.Variable(
                    tensorflow_empty_frnt(
                        (in_channels, out_channels // groups, *kernel_size),
                        **factory_kwargs,
                    )
                )
            else:
                self.weight2 = tf.Variable(
                    tensorflow_empty_frnt(
                        (out_channels, in_channels // groups, *kernel_size),
                        **factory_kwargs,
                    )
                )

    # not an ivy api so attributes should not be deleted
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    transformed = transform_function(tensorflow__ConvNd)

    assert (
        "self.weight1 = tf.Variable(tensorflow_empty_frnt((*kernel_size, out_channels // groups, in_channels), **factory_kwargs))"
        in transformed
    )
    assert (
        "self.weight2 = tf.Variable(tensorflow_empty_frnt((*kernel_size, in_channels // groups, out_channels), **factory_kwargs))"
        in transformed
    )
    del os.environ["APPLY_TRANSPOSE_OPTIMIZATION"]


if __name__ == "__main__":
    pytest.main([__file__])
