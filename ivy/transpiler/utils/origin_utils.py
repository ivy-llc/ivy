# global
from collections.abc import Sequence
import gast
import inspect

# local
from ..transformations import transformer_globals as glob

# NOTE: Please use `getattr(ast_node, ORIGI_INFO)` instead of . operation to get the original information of ast node.
ORIGI_INFO = "Original information of source code for ast node."


class Location:
    """
    Location information of source code.
    """

    __slots__ = (
        "filepath",
        "lineno",
        "col_offset",
    )

    def __init__(self, filepath, lineno, col_offset=None):
        self.filepath = filepath
        self.lineno = lineno
        self.col_offset = col_offset

    def __str__(self):
        return "location: {}:{}:{}".format(self.filepath, self.lineno, self.col_offset)

    @property
    def line_location(self):
        return (self.filepath, self.lineno)


class OriginInfo:
    """
    Original information of source code.
    """

    __slots__ = (
        "location",
        "function_name",
        "obj_like",
        "origin_obj",
        "from_global",
        "is_ivy_global",
        "global_dependencies",
        "source_code",
    )

    def __init__(
        self,
        location,
        function_name,
        source_code,
        obj_like,
        origin_obj,
        from_global=False,
    ):
        self.location = location
        self.function_name = function_name
        self.obj_like = obj_like
        # object from which this origin belongs. In normal cases, this represents the immediate parent
        # of the current object. However, in the case of globals, this represents the root object from
        # which the global statement originates.
        self.origin_obj = origin_obj
        self.source_code = source_code
        self.from_global = (
            from_global  # flag to indicate if the origin is from global stmt
        )
        self.is_ivy_global = (
            False  # flag to indicate the global_stmt represents an ivy global
        )
        self.global_dependencies = (
            {}
        )  # global_dependencies of the origin object. Used to set the global_dependencies for a GlobalObjectLike

    def __str__(self):
        return "{} \nsource_code: {}  in function {}\n  ".format(
            self.location, self.source_code, self.function_name
        )

    def formated_message(self):
        flag_for_origin_info = "(* user code *)"
        return '    File "{}", line {}, in {} {}\n\t{}'.format(
            self.location.filepath,
            self.location.lineno,
            self.function_name,
            flag_for_origin_info,
            self.source_code.lstrip(),
        )

    def as_frame(self):
        return (
            self.location.filepath,
            self.location.lineno,
            self.function_name,
            self.source_code.lstrip(),
        )


class OriginInfoAttacher(gast.NodeTransformer):
    """
    Attach original source information to AST node according corresponding function.
    """

    def __init__(self, root, obj_like, origin_object, from_global=False):
        self.root = root
        self.obj_like = obj_like
        self.origin_object = origin_object
        self.func = obj_like.get_unwrapped_object()
        self.filepath = inspect.getsourcefile(self.func)
        self.source_code = inspect.getsource(self.func)
        self.from_global = from_global
        self.current_func = []
        self.decorators = obj_like.ivy_decorators
        if self.func.__name__ == "asarray":
            existing_decorators = set(
                line.strip()
                for line in self.source_code.split("\n")
                if line.strip().startswith("@")
            )
            self.decorators = [
                dec.replace("@", "")
                for dec in self.decorators
                if dec.replace("@", "@_asarray_") not in existing_decorators
            ]

    def transform(self):
        source_lines, begin_lineno = inspect.getsourcelines(self.func)
        for decorator in reversed(self.decorators):
            source_lines.insert(0, "@" + decorator)
            begin_lineno -= 1
        begin_line = source_lines[0]
        self.col_offset = len(begin_line) - len(begin_line.lstrip())
        self.source_lines = [line.strip("\n") for line in source_lines]
        self.lineno_offset = begin_lineno - 1 - len(self.decorators)
        self.visit(self.root)

    def visit(self, node):
        if isinstance(node, (gast.FunctionDef, gast.ClassDef)):
            self.current_func.append(node)
        if hasattr(node, "lineno"):
            self._attach_origin_info(node)
        self.generic_visit(node)

        if isinstance(node, (gast.FunctionDef, gast.ClassDef)):
            self.current_func.pop()
        return node

    def _attach_origin_info(self, node):
        assert isinstance(node, gast.AST)
        assert hasattr(node, "lineno")

        lineno = self._abs_lineno(node)
        col_offset = self._abs_col_offset(node)
        loc = Location(self.filepath, lineno, col_offset)
        if self.current_func:
            func_name = self.current_func[-1].name
        else:
            func_name = None
        code_line = self.source_lines[node.lineno - 1]

        origin_info = OriginInfo(
            loc,
            func_name,
            code_line,
            obj_like=self.obj_like,
            origin_obj=self.origin_object,
            from_global=self.from_global,
        )
        setattr(node, ORIGI_INFO, origin_info)

    def _abs_lineno(self, node):
        # NOTE:
        #   There are differences in ast_node.lineno between PY3.8+ and PY3.8-.
        #   If the first gast.FunctionDef has decorator, the lineno of gast.FunctionDef is differs.
        #       1. < PY3.8
        #           its lineno equals to the lineno of the first decorator node, which is not right.
        #       2. >= PY3.8
        #           its lineno is the actual lineno, which is right.

        return self.lineno_offset + node.lineno

    def _abs_col_offset(self, node):
        return self.col_offset + node.col_offset


global_origin_info_map = {}


def create_and_update_origin_info_map(transformed_node, static_func, is_global=True):
    """
    Creates a original information map between transformed function and original function.

    Args:
        transformed_node(gast.AST): The AST node of transformed function with attached source information of original function.
        static_func(Callable): The original function transformed by Source2Source corresponding to transformed_node.

    Returns:
        The original information map.
    """

    origin_info_map = {}
    static_source = inspect.getsource(static_func)
    static_node = gast.parse(static_source)
    static_node = attach_origin_info(static_node, static_func)

    for t_node, s_node in ast_walk(transformed_node, static_node):
        assert type(t_node) == type(
            s_node
        ), "The node types should be the same, but received type(t_node) is {}, and type(s_node) is {}.".format(
            type(t_node), type(s_node)
        )
        Source2Source_info = getattr(t_node, ORIGI_INFO, None)
        static_info = getattr(s_node, ORIGI_INFO, None)

        if Source2Source_info is None or static_info is None:
            continue
        static_loc = static_info.location.line_location
        exist_origin_info = origin_info_map.get(static_loc)

        if exist_origin_info is not None:
            if exist_origin_info.location.lineno >= Source2Source_info.location.lineno:
                continue
            if (
                exist_origin_info.location.col_offset
                <= Source2Source_info.location.col_offset
            ):
                continue

        origin_info_map[static_loc] = Source2Source_info

    global_origin_info_map.update(origin_info_map)
    if is_global:
        return global_origin_info_map

    return origin_info_map


def attach_origin_info(ast_node, obj_like, origin_object, from_global=False):
    """
    Attach original source information to AST node according corresponding function.

    Args:
        ast_node(gast.AST): The AST node to attach original source information.
        obj_like(FuncLikeObj/TypeLikeObj): The corresponding object of ast_node. Parse the original information from this object.
        origin_object(FuncLikeObj/TypeLikeObj):The parent object of obj_like
    Returns:
        An AST node attached original source information.
    """
    resolver = OriginInfoAttacher(
        ast_node,
        obj_like=obj_like,
        origin_object=origin_object,
        from_global=from_global,
    )
    resolver.transform()
    return ast_node


def ast_walk(transformed_node, static_node):
    """
    Recursively yield all descendant nodes in the trees starting at transformed_node and static_node (including itself) in parallel.

    NOTE:
        Function ast.walk is not used because it yield all descendant nodes in no specified order.
    """

    def _as_list(x):
        if x is None:
            return []
        return list(x) if isinstance(x, Sequence) else [x]

    transformed_node_list = _as_list(transformed_node)
    static_node_list = _as_list(static_node)

    while transformed_node_list:
        assert len(transformed_node_list) == len(static_node_list)
        t_node = transformed_node_list.pop()
        s_node = static_node_list.pop()
        if type(t_node) != type(s_node):
            # NOTE:
            # Node types should be strictly required, but there is no strict distinction between gast.Load and gast.Param
            # in the ast transformation process.
            if isinstance(t_node, (gast.Load, gast.Param)) or isinstance(
                s_node, (gast.Load, gast.Param)
            ):
                continue

        assert type(t_node) == type(
            s_node
        ), "The node types should be the same, but received type(t_node) is {}, and type(s_node) is {}.".format(
            type(t_node), type(s_node)
        )

        yield t_node, s_node

        for field in t_node._fields:
            t_node_child = getattr(t_node, field)
            s_node_child = getattr(s_node, field)

            if isinstance(t_node_child, gast.AST):
                transformed_node_list.append(t_node_child)
                static_node_list.append(s_node_child)
            elif isinstance(t_node_child, (list, tuple)):
                assert len(t_node_child) == len(s_node_child)
                for d_item, s_item in zip(t_node_child, s_node_child):
                    if isinstance(d_item, gast.AST):
                        transformed_node_list.append(d_item)
                        static_node_list.append(s_item)
