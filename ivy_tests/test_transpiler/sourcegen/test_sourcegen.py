import gast
import textwrap
from ivy.transpiler.utils.ast_utils import (
    TranslatedContext,
    TranslatedFunctionVisitor,
    extract_target_object_name,
    reorder_objects,
)
from ivy.transpiler.utils.naming_utils import NAME_GENERATOR
from ivy.transpiler.utils.logging_utils import Logger


def test_TranslatedFunctionVisitor_fn():
    fn_test_code = textwrap.dedent(
        """
    @tensorflow_function_decorator
    def test_function(tensorflow_param: tensorflow_param_type = tensorflow_default) -> tensorflow_return_type:
        return tensorflow_result
    """
    )
    ast_root = gast.parse(fn_test_code)
    visitor = TranslatedFunctionVisitor()
    NAME_GENERATOR.set_prefixes(target="tensorflow")
    visitor.visit(ast_root)

    expected_results = {
        "tensorflow_function_decorator": TranslatedContext.DECORATOR,
        "tensorflow_param": TranslatedContext.FUNCTION_ARGS,
        "tensorflow_param_type": TranslatedContext.TYPE_SPEC,
        "tensorflow_default": TranslatedContext.FUNCTION_ARGS,
        "tensorflow_return_type": TranslatedContext.TYPE_SPEC,
        "tensorflow_result": TranslatedContext.VARIABLE,
    }
    print(visitor.translated_nodes)
    assert set(visitor.translated_nodes.keys()) == set(
        expected_results.keys()
    ), "Mismatched keys in translated_nodes"
    for key, expected_value in expected_results.items():
        assert (
            visitor.translated_nodes[key] == expected_value
        ), f"Mismatched value for {key}"


def test_TranslatedFunctionVisitor_cls():
    class_test_code = textwrap.dedent(
        """
    @tensorflow_decorator
    class TestClass(tensorflow_base_class[tensorflow_base_class2], metaclass=tensorflow_metaclass):
        tensorflow_class_attribute: tensorflow_type = tensorflow_value

        @tensorflow_method_decorator
        def test_method(self, 
                        arg1: tensorflow_type1, 
                        arg2: tensorflow_type2 = tensorflow_default_value,
                        *args: tensorflow_args_type,
                        **kwargs: tensorflow_kwargs_type) -> tensorflow_return_type:
            tensorflow_local_var = tensorflow_function(tensorflow_arg)
            return tensorflow_result
    """
    )
    ast_root = gast.parse(class_test_code)
    visitor = TranslatedFunctionVisitor()
    NAME_GENERATOR.set_prefixes(target="tensorflow")
    visitor.visit(ast_root)

    expected_results = {
        "tensorflow_decorator": TranslatedContext.DECORATOR,
        "tensorflow_base_class": TranslatedContext.BASE,
        "tensorflow_base_class2": TranslatedContext.BASE,
        "tensorflow_metaclass": TranslatedContext.BASE,
        "tensorflow_class_attribute": TranslatedContext.CLASS_ATTRIBUTE,
        "tensorflow_type": TranslatedContext.TYPE_SPEC,
        "tensorflow_value": TranslatedContext.CLASS_ATTRIBUTE,
        "tensorflow_method_decorator": TranslatedContext.DECORATOR,
        "tensorflow_type1": TranslatedContext.TYPE_SPEC,
        "tensorflow_type2": TranslatedContext.TYPE_SPEC,
        "tensorflow_default_value": TranslatedContext.FUNCTION_ARGS,
        "tensorflow_args_type": TranslatedContext.TYPE_SPEC,
        "tensorflow_kwargs_type": TranslatedContext.TYPE_SPEC,
        "tensorflow_return_type": TranslatedContext.TYPE_SPEC,
        "tensorflow_local_var": TranslatedContext.VARIABLE,
        "tensorflow_function": TranslatedContext.VARIABLE,
        "tensorflow_arg": TranslatedContext.VARIABLE,
        "tensorflow_result": TranslatedContext.VARIABLE,
    }

    assert set(visitor.translated_nodes.keys()) == set(
        expected_results.keys()
    ), "Mismatched keys in translated_nodes"
    for key, expected_value in expected_results.items():
        assert (
            visitor.translated_nodes[key] == expected_value
        ), f"Mismatched value for {key}"


def test_extract_target_object_name():
    NAME_GENERATOR.reset_state()
    NAME_GENERATOR.new_prefix = "new_"
    NAME_GENERATOR.old_prefix = "old_"

    test_cases = [
        ("new_object_bknd_base_count_1", "object"),
        ("old_object_frnt", "object"),
        ("object_base_count_2", "object"),
        ("new_function_bknd_base_count_3", "function"),
        ("old_function_frnt_base_count_4", "function"),
        ("simple_name", "simple_name"),
    ]

    for name, expected in test_cases:
        result = extract_target_object_name(name)
        assert (
            result == expected
        ), f"Failed for {name}: expected {expected}, got {result}"


def normalize_code(code):
    """Helper function to normalize the code by stripping off new lines and whitespaces."""
    return code.replace(" ", "").replace("\n", "")


def test_reorder_objects_with_matches():

    # Simple source and target modules for testing
    source_module_code = textwrap.dedent(
        """
        import os
        class A: pass
        def foo(): pass
        """
    )
    target_module_code = textwrap.dedent(
        """
        import os
        def foo(): pass
        class A: pass
        """
    )

    # Expected reordered target module
    expected_reordered_code = textwrap.dedent(
        """
        import os
        class A: pass
        def foo(): pass
        """
    )

    # Call the function
    logger = Logger()
    result = textwrap.dedent(
        reorder_objects(source_module_code, target_module_code, logger)
    )

    # Assertions
    assert normalize_code(result) == normalize_code(
        expected_reordered_code
    ), "Reordering failed"


def test_reorder_objects_with_mismatches():

    # Source and target modules with additional mismatched objects in target
    source_module_code = textwrap.dedent(
        """
        import os
        class A: pass
        def foo(): pass
        GLOB = baz()
        """
    )
    target_module_code = textwrap.dedent(
        """
        import os
        tf.experimental.numpy.experimental_enable_numpy_behavior(True)
        def bar(): pass
        def foo(): pass
        class A: pass
        GLOB = baz()
        """
    )

    # Expected reordered target module with `bar` function added at the end
    expected_reordered_code = textwrap.dedent(
        """
        import os
        tf.experimental.numpy.experimental_enable_numpy_behavior(True)
        class A: pass
        def foo(): pass
        GLOB = baz()
        def bar(): pass
        """
    )

    # Call the function
    logger = Logger()
    result = reorder_objects(source_module_code, target_module_code, logger)
    # Assertions
    print("result", result)
    print("expected_reordered_code", expected_reordered_code)
    assert normalize_code(result) == normalize_code(
        expected_reordered_code
    ), "Reordering with mismatches failed"
