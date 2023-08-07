from ivy_tests.test_ivy.helpers.pipeline_helper import update_backend


def partition_function_tree(fn_tree: str):
    module_tree, _, fn_name = fn_tree.rpartition(".")
    return module_tree, fn_name


def partition_method_tree(method_tree: str):
    module_class_tree, _, method_name = method_tree.rpartition(".")
    module_tree, class_name = module_class_tree
    return module_tree, class_name, method_name


def import_function_from_ivy(module_tree: str, fn_name: str, framework: str):
    with update_backend(framework) as ivy_backend:
        module = ivy_backend.utils.dynamic_import.import_module(module_tree)
        fn = getattr(module, fn_name)
        return fn


def import_method_from_ivy(
    module_tree: str, class_name: str, method_name: str, framework: str
):
    with update_backend(framework) as ivy_backend:
        module = ivy_backend.utils.dynamic_import.import_module(module_tree)
        cls = getattr(module, class_name)
        method = getattr(cls, method_name)
        return method
