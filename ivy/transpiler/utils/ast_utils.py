import ast
import astor
import gast
import importlib
import inspect
import re
from typing import List

from .naming_utils import NAME_GENERATOR


def ast_to_source_code(ast_node):
    """
    Transforms ast node into source code.
    """
    if isinstance(ast_node, str):
        return ast_node

    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s."
            % type(ast_node)
        )
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)

    # Do not wrap lines even if they are too long
    def pretty_source(source):
        return "".join(source)

    source_code = astor.to_source(ast_node, pretty_source=pretty_source)
    return source_code


def check_syntax(source_code: str) -> bool:
    """
    Check if the provided Python source code is syntactically correct.

    Args:
        source_code (str): The source code to check.

    Returns:
        bool: True if the code is syntactically correct, False otherwise.
    """
    try:
        # Attempt to compile the source code to check for syntax errors
        compile(source_code, "<string>", "exec")
        return True
    except SyntaxError as e:
        raise SyntaxError(f"Syntax Error: {e}")


def parse_source_code(module):
    try:
        source_code = inspect.getsource(module)
        return gast.parse(source_code)
    except (TypeError, OSError):
        return None


def replace_placeholders(
    input_node: gast.AST, variable_nodes: List[gast.AST], placeholder="_"
) -> gast.AST:
    """
    Use this function to replace a recurring `placeholder`
    in an `input_node` stringified representation with all the
    stringified representations from the list of `variable_nodes` provided.
    """
    input_string = ast_to_source_code(input_node).strip()
    variable_names = [ast_to_source_code(node).strip() for node in variable_nodes]

    var_index = 0

    # Convert the string to a list for easy modification
    string_list = list(input_string)

    for i, char in enumerate(input_string):
        if char == placeholder:
            if var_index < len(variable_names):
                # Replace '_' with the next variable name
                string_list[i] = variable_names[var_index]
                var_index += 1
            else:
                # If the list of variable names is exhausted, break the loop
                break

    # Join the list back into a string
    result_string = "".join(string_list).replace("_", "'_'")
    return gast.parse(result_string).body[0].value


def set_parents(node, parent=None):
    """Assign parents for each node in a gast tree."""
    node.parent = parent
    for child in gast.iter_child_nodes(node):
        set_parents(child, node)

def extract_target_object_name(name):
    """
    Extracts the target object name by removing specific prefixes and suffixes from the input name.

    Args:
        name (str): The input name from which to extract the target object name.

    Returns:
        str: The cleaned name with prefixes and suffixes removed.

    This function:
        - Removes the prefix if it matches the `new_prefix` or `old_prefix` from `NAME_GENERATOR`.
        - Removes specific suffixes like "_bknd", "_frnt", and numeric suffixes (e.g., "base_count_1", "base_count_2").
    """

    def remove_prefix(s):
        if s.startswith(NAME_GENERATOR.new_prefix):
            return s[len(NAME_GENERATOR.new_prefix) :]
        elif s.startswith(NAME_GENERATOR.old_prefix):
            return s[len(NAME_GENERATOR.old_prefix) :]
        else:
            return s

    def remove_suffix(s):
        # Remove numeric suffix(eg: _base_count_1, _base_count_2 etc.) if present
        s = re.sub(r"(_base_count_\d)+$", "", s)
        # Remove frontend/backend suffix
        pattern = r"_bknd_|_bknd|_frnt_|_frnt"
        s = re.sub(pattern, "", s)
        return s

    return remove_suffix(remove_prefix(name))


def get_module(name, package=None):
    try:
        if package:
            name = package + "." + name
        return importlib.import_module(name)
    except ImportError:
        return None


def keyword_in_keyword_args(node, keyword: str):
    return any(kw.arg == keyword for kw in node.keywords)


def is_super_call_node(node):
    """
    Check if a node represents a call to super.

    """
    _node = node if not isinstance(node, ast.AST) else gast.ast_to_gast(node)
    return (
        isinstance(_node, gast.Attribute)
        and isinstance(_node.value, gast.Call)
        and isinstance(_node.value.func, gast.Name)
        and _node.value.func.id == "super"
    )


def get_attribute_full_name(node):
    assert isinstance(
        node, gast.Attribute
    ), "Input non-Attribute node to get attribute full name"
    return astor.to_source(gast.gast_to_ast(node)).strip()


def property_to_func(orig_obj, node):
    """
    Retrieves the function (getter, setter, or deleter) associated with a Python property
    based on the context of use.
    """
    # Determine the function associated with the property based on the context
    if isinstance(node.ctx, gast.Load):
        property_func = orig_obj.fget
    elif isinstance(node.ctx, gast.Store):
        property_func = orig_obj.fset
    else:
        property_func = orig_obj.fdel

    return property_func


def is_builtin_type_call_node(node):
    assert isinstance(node, gast.Call), "Input non-Call node for is_builtin_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        func = func_str.strip().split(".")[-1]
        return any([func in dir(cls) for cls in (list, dict, set, str)])
    except Exception:
        return False


def is_unpacking_assignment(node):
    if not isinstance(node, gast.Assign):
        return False

    if len(node.targets) != 1 or not isinstance(node.targets[0], gast.Tuple):
        return False

    # Check if any element in the tuple is a Name (variable)
    if any(isinstance(elt, gast.Name) for elt in node.targets[0].elts):
        # Check if the value being assigned is a Name (variable)
        if isinstance(node.value, gast.Name):
            return True

        # Check if the value being assigned is an attribute (e.g., object.attribute)
        elif isinstance(node.value, gast.Attribute):
            return True

        # Check if the value being assigned is a Subscript (e.g., list[index])
        elif isinstance(node.value, gast.Subscript):
            return True

    return False
