import ast
import inspect

replace_map = {}


def replace_with(new_func):
    """Decorate a function/method/attribute to be replaced by another.

    Parameters
    ----------
    new_func
        The function that will replace the original.
    """

    def decorator(original_func):
        if not callable(original_func) or not callable(new_func):
            raise TypeError(
                f"Both '{original_func.__name__}' and '{new_func.__name__}' should be"
                " callable."
            )

        if inspect.getfullargspec(original_func) != inspect.getfullargspec(new_func):
            raise ValueError(
                f"Replacement function '{new_func.__name__}' arguments don't match"
                f" '{original_func.__name__}' arguments."
            )

        new_func_name = f"{original_func.__name__}_replacement"

        if new_func_name in globals():
            raise NameError(
                f"Name '{new_func_name}' already exists in global namespace."
            )

        globals()[new_func_name] = new_func
        replace_map[original_func.__name__] = new_func_name
        return original_func

    return decorator


class ReplaceFunction(ast.NodeTransformer):
    """AST Node Transformer to replace function calls, methods, and
    attributes."""

    def visit_Attribute(self, node):
        if (
            isinstance(node.value, ast.Name)
            and f"{node.value.id}.{node.attr}" in replace_map
        ):
            return ast.copy_location(
                ast.Name(replace_map[f"{node.value.id}.{node.attr}"], node.ctx), node
            )
        return node

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and f"{node.func.value.id}.{node.func.attr}" in replace_map
        ):
            node.func = ast.Name(
                replace_map[f"{node.func.value.id}.{node.func.attr}"], node.func.ctx
            )
        elif isinstance(node.func, ast.Name) and node.func.id in replace_map:
            node.func.id = replace_map[node.func.id]
        return node


def transform_function(func):
    """Transform the function by replacing its calls based on the
    replace_map."""
    source = inspect.getsource(func)
    tree = ast.parse(source)
    transformed_tree = ReplaceFunction().visit(tree)
    transformed_code = ast.unparse(transformed_tree)

    namespace = {}
    exec(transformed_code, globals(), namespace)
    return namespace[func.__name__]
