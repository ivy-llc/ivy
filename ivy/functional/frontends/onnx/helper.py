from ivy.functional.frontends.onnx.proto import NodeProto

from ivy_tests.test_ivy.helpers.testing_helpers import _import_fn


def make_node(
    op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs
):
    # keep things upper case to follow ONNX naming convention
    fn_tree = "ivy.functional.frontends.onnx." + op_type
    callable_fn, fn_name, fn_mod = _import_fn(fn_tree)

    node = NodeProto()
    node._fn = callable_fn
    node._fn_mod = fn_mod
    node._fn_name = fn_name
    node.input = inputs
    node.output = outputs
    node.name = name

    return node
