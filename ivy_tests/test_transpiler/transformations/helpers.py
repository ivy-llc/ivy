def load_fn_from_str(source_str, fn_name):
    """Executes the source code passed as arguments and returns the defined "fn" """
    namespace = {}
    exec(source_str, namespace)
    fn = namespace[fn_name]
    return fn
