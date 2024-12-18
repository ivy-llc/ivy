# global
import gast
import ivy

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ....utils.ast_utils import (
    ast_to_source_code,
)
from ....utils.api_utils import (
    get_native_array_str_from_backend,
    get_native_module_str_from_backend,
)
from ....utils.naming_utils import NAME_GENERATOR
from .ivy_postprocessing_transformer import (
    IvyCodePostProcessor,
)


class IvyToTFCodePostProcessor(IvyCodePostProcessor):
    """
    Peform post-processing for TensorFlow backend.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
        new_name="tensor",
    ) -> None:
        super().__init__(root, transformer, configuration, new_name=new_name)
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

    def _handle_ivy_array(self, node):
        new_name = get_native_array_str_from_backend(ivy.backend)
        return gast.parse(f"{ivy.backend}.{new_name}").body[0].value

    def _handle_ivy_variable(self, node):
        try:
            import keras 
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Keras is required for TensorFlow backend."
            )
        variable_name = "tensorflow.keras.Variable" if keras.__version__ >= "3.0.0" else "tensorflow.Variable"
        return gast.parse(variable_name).body[0].value

    def _handle_ivy_module(self, node):
        new_name = get_native_module_str_from_backend(
            backend_str=ivy.backend,
            is_root_obj=self.transformer.object_like.is_root_obj,
            depth=self.transformer.object_like.depth,
        )
        new_name = new_name.replace(".", "_")
        return gast.parse(f"{new_name}").body[0].value

    def _handle_assign_transform(self, node):
        try:
            import keras 
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Keras is required for TensorFlow backend."
            )
        variable_name = "tensorflow.keras.Variable" if keras.__version__ >= "3.0.0" else "tensorflow.Variable"
        return gast.Call(
            func=gast.parse(variable_name).body[0].value,
            args=node.value.args,
            keywords=node.value.keywords,
        )

    def _handle_tf_name_scope(self, node):
        stmt_value = ast_to_source_code(node.value)

        if any(
            substr in stmt_value
            for substr in ("ivy.Array", "tensorflow.Variable", "tensorflow_")
        ):
            if isinstance(node.targets[0], gast.Name):
                name = node.targets[0].id
            else:
                name = "".join(
                    ast_to_source_code(node.targets[0]).split(".")[1:]
                ).strip()

            if self.transformer.object_like.is_root_obj:
                name = (
                    NAME_GENERATOR.get_name(self.transformer.object_like) + "/" + name
                )

            name_scope = gast.Constant(
                value=name,
                kind=None,
            )
            var_created = False
            for n in gast.walk(node.value):
                if isinstance(n, gast.Call) and ast_to_source_code(n.func).strip() in (
                    "tensorflow.Variable",
                    "ivy.Array",
                ):
                    keyword = gast.keyword(arg="name", value=name_scope)
                    n.keywords.append(keyword)
                    var_created = True
            if not var_created:
                with_node = self._create_with_node(
                    with_body=node, name_scope_node=name_scope
                )
                return with_node

    def _transform_isinstance_check(self, node):
        """
        if not isinstance(module, tensorflow_keras_Layer) --> if not isinstance(module, (tensorflow_keras_Layer, tf.keras.layers.Layer))
        """
        new_args = [
            node.args[0],
            gast.Tuple(
                elts=[
                    node.args[1],
                    gast.parse("tensorflow.keras.layers.Layer").body[0].value,
                ],
                ctx=gast.Load(),
            ),
        ]
        node.args = new_args
        return node

    def _get_forward_name(self, node):
        return "call"

    def _maybe_convert_device_attribute(self, node):
        return node

    def _maybe_add_unpack_inputs_decorator(self, node):
        # add the `unpack_inputs` decorator to the call method
        node.decorator_list.append(
            gast.Name(
                id="unpack_inputs",
                ctx=gast.Load(),
                type_comment=None,
                annotation=None,
            )
        )
        self.transformer._from_imports.add(
            ("transformers.modeling_tf_utils", "unpack_inputs", None)
        )
        return node

    def _maybe_add_tf_name_scope(self, node):
        # TODO: add this back, ensuring name_scope is only added where needed (so it doesn't break tf.function for other kornia/hf examples)
        # if self.transformer.object_like.type == Types.ClassType:
        #     self._in_method = True
        #     self.generic_visit(node)
        #     self._in_method = False
        return node

    def _maybe_modify_add_module(self, node):
        # inject the below-mentioned stmt after the line: self._modules[name] = module
        # this is done in order to attach numeric prefix to the weights created
        # inside inner submodules (eg:blocks/1/...,blocks/2/...)

        # TODO: re-instantiating the class slows down the model's initialization
        # significantly. Try to figure out an alternative solution. Eg: adding
        # the tf.name_scope ctx manager directly inside the for-loop/list_comp
        """
        with tensorflow.name_scope(name):
            module = module.from_config(module.get_config())
        """
        return node
        body_node = gast.parse("module = module.from_config(module.get_config())").body[
            0
        ]
        name_scope = gast.Name(
            id="name", ctx=gast.Load(), type_comment=None, annotation=None
        )
        with_node = self._create_with_node(
            with_body=body_node, name_scope_node=name_scope
        )
        # Find the index of the 'self._modules[name] = module' stmt
        for i, stmt in enumerate(node.body):
            if (
                isinstance(stmt, gast.Assign)
                and ast_to_source_code(stmt.targets[0]).strip() == "self._modules[name]"
            ):
                break
        else:
            raise ValueError(
                "'self._modules[name] = module' stmt not found in function body"
            )

        # Inject the new statement right before the 'self._modules[name] = module' stmt
        node.body[i:i] = [with_node]
        return node

    def _maybe_replace_with_native_array_calls(self, node):
        func_str = ast_to_source_code(node.func).strip()
        if func_str in ("tensorflow.Tensor", "Tensor", "tf.Tensor", "ivy.Array"):
            new_func = gast.Attribute(
                value=gast.Name(
                    id="tensorflow",
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None,
                ),
                attr="convert_to_tensor",
                ctx=gast.Load(),
            )
            node.func = gast.fix_missing_locations(new_func)
        return node

    def _create_with_node(self, with_body, name_scope_node):
        # Create a gast.With node with a tensorflow.name_scope context manager.
        # This allows TF to create variables with names prefixed with the given name_scope.
        with_node = gast.With(
            items=[
                gast.withitem(
                    context_expr=gast.Call(
                        func=gast.Attribute(
                            value=gast.Name(
                                id="tensorflow",
                                ctx=gast.Load(),
                                annotation=None,
                                type_comment=None,
                            ),
                            attr="name_scope",
                            ctx=gast.Load(),
                        ),
                        args=[name_scope_node],
                        keywords=[],
                    ),
                    optional_vars=None,
                )
            ],
            body=[with_body],
            type_comment=None,
        )
        return with_node

    def _replace_ivy_array_pattern(self, elts):
        """
        Transform the type check argument of an isinstance call
        to replace any occurrence of (ivy.Array, ivy.Array) with
        (tensorflow.Tensor, tensorflow.Variable).
        """
        # Pattern to look for: (ivy.Array, ivy.Array)
        pattern = [
            gast.Attribute(
                value=gast.Name(id="ivy", ctx=gast.Load()),
                attr="Array",
                ctx=gast.Load(),
            ),
            gast.Attribute(
                value=gast.Name(id="ivy", ctx=gast.Load()),
                attr="Array",
                ctx=gast.Load(),
            ),
        ]

        # Serialize the pattern into a string
        pattern_dump = [gast.dump(node) for node in pattern]

        # Traverse through the elements and replace any matching pattern
        transformed_elts = []
        i = 0
        while i < len(elts):
            # Serialize current slice of elements and compare with pattern_dump
            elts_dump = [gast.dump(node) for node in elts[i : i + 2]]
            if elts_dump == pattern_dump:  # Check if we found the pattern
                # Replace the pattern with (tensorflow.Tensor, tensorflow.Variable)
                transformed_elts.extend(
                    [
                        gast.Attribute(
                            value=gast.Name(id="tensorflow", ctx=gast.Load()),
                            attr="Tensor",
                            ctx=gast.Load(),
                        ),
                        gast.Attribute(
                            value=gast.Name(id="tensorflow", ctx=gast.Load()),
                            attr="Variable",
                            ctx=gast.Load(),
                        ),
                    ]
                )
                i += 2  # Skip the matched elements
            else:
                transformed_elts.append(elts[i])
                i += 1

        return transformed_elts

    def _maybe_modify_inplace_update_fn(self, node):
        # Check if the function name contains "inplace_update"
        if "inplace_update" in node.name:
            # Step 1: Modify the default value of keep_input_dtype to True
            self._modify_keep_input_dtype_kwarg(node)

            # Step 2: Modify assignment nodes to use val_native on the right-hand side
            self._modify_assignments_to_val_native(node)

            # Step 3: Replace `if tensorflow_is_ivy_array_bknd(x)` with `x = x_native`
            self._replace_if_tensorflow_is_ivy_array_bknd(node)

        return node

    def _modify_keep_input_dtype_kwarg(self, node):
        """Step 1: Modify keep_input_dtype kwarg default value to True in inplace update signature."""
        for kwarg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if ast_to_source_code(kwarg).strip() == "keep_input_dtype":
                # Modify default value to True if it exists
                if isinstance(default, gast.Constant):
                    default.value = True
                    break

    def _modify_assignments_to_val_native(self, node):
        """Step 2: Modify assignment nodes to use val_native on the RHS in inplace_update body."""

        class AssignVisitor(gast.NodeTransformer):
            def visit_Assign(self, assign_node):
                for target in assign_node.targets:
                    if ast_to_source_code(target).strip() == "x":
                        # Modify the right-hand side to use val_native (keep function calls)
                        val_native_node = gast.Name(id="val_native", ctx=gast.Load())
                        # If the right-hand side is a function call, replace its first argument with "val_native"
                        if isinstance(assign_node.value, gast.Call):
                            assign_node.value.args[0] = val_native_node
                        else:
                            # Otherwise, replace the entire right-hand side with "val_native"
                            assign_node.value = val_native_node
                self.generic_visit(assign_node)
                return assign_node

        AssignVisitor().visit(node)

    def _replace_if_tensorflow_is_ivy_array_bknd(self, node):
        """Step 3: Replace `if tensorflow_is_ivy_array_bknd(x)` with `x = x_native` in inplace_update body."""

        class IfVisitor(gast.NodeTransformer):
            def visit_If(self, if_node):
                if isinstance(if_node.test, gast.Call):
                    func = if_node.test.func
                    if (
                        ast_to_source_code(func).strip()
                        == "tensorflow_is_ivy_array_bknd"
                    ):
                        # Replace entire if-else block with `x = x_native`
                        return gast.Assign(
                            targets=[gast.Name(id="x", ctx=gast.Store())],
                            value=gast.Name(id="x_native", ctx=gast.Load()),
                        )
                self.generic_visit(if_node)
                return if_node

        IfVisitor().visit(node)
