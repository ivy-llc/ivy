import gast

from ivy.transpiler.utils.ast_utils import ast_to_source_code

from .visitors import ObjectOrderVisitor


def get_object_order(module_ast, return_as_dict=True):
    """
    Extracts and returns the order of class, function, and global variable definitions
    from the given AST of a module.

    Args:
        module_ast (gast.AST): The root of the AST representing the module.
        return_as_dict (bool): If True, returns a dictionary where the keys are tuples
                               describing the type ("class", "function", or "global") and
                               the object name, and the values are the corresponding AST nodes.
                               If False, returns a list of the object keys (type, name).

    Returns:
        Union[Dict[Tuple[str, str], gast.AST], List[Tuple[str, str]]]:
            Either a dictionary with object type and name as keys and AST nodes as values,
            or a list of the object type-name tuples.

    Example:
        >>> import gast
        >>> source_code = '''
        ... class MyClass:
        ...     def my_method(self):
        ...         pass
        ...
        ... def my_function():
        ...     pass
        ... '''
        >>> module_ast = gast.parse(source_code)
        >>> object_order = get_object_order(module_ast, return_as_dict=True)
        >>> print(object_order.keys())
        dict_keys([('class', 'MyClass'), ('function', 'my_function')])
    """
    visitor = ObjectOrderVisitor()
    visitor.visit(module_ast)
    if not return_as_dict:
        return list(visitor.order.keys())
    return visitor.order


def reorder_objects(source_module, target_module, logger):
    """
    Reorders objects in the target module to match the order of objects in the source module.

    Args:
        source_module (str): Source module's code as a string to derive object order.
        target_module (str): Target module's code as a string to be reordered.
        logger (Logger): Logger instance used to log any mismatches between the source and target modules.

    Returns:
        str: The reordered target module code as a string.

    The function:
        - Parses the source and target modules into ASTs.
        - Extracts the order of objects (classes, functions, etc.) from the source module.
        - Reorders the objects in the target module to match the source module.
        - Logs any objects in the target module that don't match the source.
        - Returns the reordered target module as source code.
    """
    assert isinstance(source_module, str), "source_module must be a string"
    assert isinstance(target_module, str), "target_module must be a string"

    source_ast = gast.parse(source_module)
    target_ast = gast.parse(target_module)

    # Get the order of objects from the source module
    source_order = get_object_order(source_ast, return_as_dict=False)
    target_objects = get_object_order(target_ast)

    experimental_node = None
    for key, value in target_objects.items():
        if "tf.experimental.numpy.experimental_enable_numpy_behavior" in key:
            experimental_node = key
            break
    # Separate imports from the rest of the code
    imports = [
        node
        for node in target_ast.body
        if isinstance(node, (gast.Import, gast.ImportFrom))
    ]

    # Reorder objects based on source_order
    reordered_body = imports[:]
    if experimental_node:
        reordered_body.append(target_objects.pop(experimental_node))
    for obj_type, obj_name in source_order:
        key = (obj_type, obj_name)
        if key in target_objects:
            reordered_body.append(target_objects.pop(key))

    if target_objects:
        # If there are any remaining objects in target_objects, add them to the end
        key_mismatches = list(target_objects.keys())
        logger.warn(f"Not all objects were reordered: {key_mismatches}")
        reordered_body.extend(target_objects.values())

    # Create a new AST with reordered body
    new_target_ast = gast.Module(body=reordered_body, type_ignores=[])

    # Convert the AST back to source code
    new_target_code = ast_to_source_code(new_target_ast)

    return new_target_code


def reorder_module_objects(source_file_path, target_file_path, logger):

    assert isinstance(source_file_path, str) and source_file_path.endswith(
        ".py"
    ), "source_file_path must be a string ending with '.py'"
    assert isinstance(target_file_path, str) and target_file_path.endswith(
        ".py"
    ), "target_file_path must be a string ending with '.py'"

    with open(source_file_path, "r", encoding="utf-8", newline="\n") as source_file:
        source_code = source_file.read()

    with open(target_file_path, "r", encoding="utf-8", newline="\n") as target_file:
        target_code = target_file.read()

    reordered_code = reorder_objects(source_code, target_code, logger)

    with open(target_file_path, "w", encoding="utf-8", newline="\n") as target_file:
        target_file.write(reordered_code)
