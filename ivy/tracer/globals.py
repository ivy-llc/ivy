"""Global variables which are used throughout the tracer-transpiler repository"""

import sys
import os

# controls when the tracing logic in _tracing_function (wrapping.py) will be executed
# and consequently whether functions will be traced.
# set to True until directly before executing the callable we want to trace
tracing_paused = True

use_reloader = True

# defines whether the tracer is currently dealing with the model in 'train'
# mode or 'eval' mode while tracing/transpiling a trainable module
current_trace_mode = "eval"

tracing_subgraph = False
subgraph_id_to_fn = dict()

# stack of function names that are currently in the process of being traced
# (used so we will not trace any functions while we are in the middle of tracing
# the current function - unless the current function is a vmap or scalar_fn)
tracing_stack = list()

# stores all the iterator chains - an iterator chain being chained
# __iter__ -> __next__ -> __next__ -> ... calls, which we need to handle differently
# because the output of the previous node is not the input to the next node
iterator_chains = dict()

# stores how ids map to weakly referenced native array objects, so we can
# create a unique id when the weakly referenced object has already been destroyed.
# this prevents potentially reusing the same id for multiple objects, which
# can occur with cpython when objects do not have overlapping lifespans
raw_id_to_weakref = dict()

# stores any unique ids (mapped from the original raw id) which have been generated
# due to id duplication in the graph
raw_id_to_unique_id = {"train": dict(), "eval": dict()}

# stores instances of classes in their source framework for transpilation purposes
class_instances = dict()

trace_classes = True

# stores transformed callables for restoration after compilation
transformed_callables = list()

# stores the ids of array objects which depend on the input parameters
# (and so define that the function using them can form part of the traced graph)
dependent_ids = {"train": set(), "eval": set()}
subgraph_dependent_ids = set()

returns_cache = list()
wrapped_fns = dict()

# for finding the problematic node when the frontend and
# original graphs aren't equivalent during transpiling
check_frontend_vs_original_differences = False

# for looking at how many nodes each frontend contributes
# to the transpiled graph and comparing original vs transpiled
# timings - useful for finding inefficient frontend/ivy implementations
check_transpiler_overhead = False
# Use the following environment variable to keep track of this flag
os.environ["CHECK_TRANSPILER_OVERHEAD"] = "False"
node_expansion = {}  # {original_node: [transpiled_node(s)]}
current_frontend = None

times = (
    {}
)  # {original_node: [original_time, transpiled_time, n x slower, transpiled_time_breakdown]}
# eg {torch.Tensor.__iadd__: [1.5e-4, 1.8e-3, 12, {'jax.numpy.asarray': 8.1e-5, 'jax.numpy.add': 1.8e-4, 'jax.numpy.copy': 1.4e-3}}
# ordered by biggest relative slowdown to smallest
transpiled_times = {}

# for user stat and api call logging
connector = None
is_unify = False
is_transpile = False


# versioning #
# ---------- #


def get_version(fw: str):
    """Helper method to get the installed version of a given framework"""
    val = sys.modules[fw].__version__ if sys.modules.get(fw, None) else None
    # do some preprocessing, like check for a + since torch adds that
    if "+" in val:
        val = val.split("+")[0]
    return val


def version_resolver(backend_version, dic: dict):
    """Helper method to retrieve the correct list from the dict depending on framework version"""
    if isinstance(backend_version, tuple):
        backend_version = backend_version[0]
    version_parts = backend_version.split(".")
    version_tuple = []
    for part in version_parts:
        if ".dev" in part:
            part = part.split(".dev")[0]
        if ".post" in part:
            part = part.split(".post")[0]
        version_tuple.append(int(part))
    version_tuple = tuple(version_tuple)

    for key in dic.keys():
        kl = key.split(" ")
        k1 = kl[0]
        if ".dev" in k1:
            k1 = k1.split(".dev")[0]
        k1 = tuple(map(int, k1.split(".")))

        if "above" in key and k1 <= version_tuple:
            return dic[key]

        if "below" in key and k1 >= version_tuple:
            return dic[key]

        if "to" in key:
            k2 = kl[2]
            if ".dev" in k2:
                k2 = k2.split(".dev")[0]
            if ".post" in k2:
                k2 = k2.split(".post")[0]
            k2 = tuple(map(int, k2.split(".")))
            if k1 <= version_tuple <= k2:
                return dic[key]


# Wrapping #
# -------- #

# tf has many functions available in multiple modules, so the modules are ordered from most to least obscure
# to ensure the most simple path is wrapped last, and hence the one we store
MODULES_TO_WRAP = {
    "numpy": [
        "numpy.lib",
        "numpy.lib.stride_tricks",
        "numpy",
        "numpy.linalg",
        "numpy.random",
        "numpy.fft",
        "numpy.polynomial",
        "scipy.fft",
        "scipy.linalg",
        "scipy.signal",
        "scipy.stats",
    ],
    "paddle": ["paddle", "paddle.linalg", "paddle.nn.functional", "paddle.fft"],
    "jax": [
        "jax",
        "jax.nn",
        "jax.lax",
        "jax.numpy.fft",
        "jax.numpy",
        "jax.numpy.linalg",
        "jax.numpy.linalg.fft",
        "jax.random",
        "jax.image",
        "jax.scipy.special",
        "jax.scipy.linalg",
        "jax.scipy.signal",
        "jax.scipy.stats",
    ],
    "tensorflow": [
        "tensorflow.raw_ops",
        "tensorflow.compat.v2.compat.v1",
        "tensorflow.compat.v1",
        "tensorflow.compat.v1.nn",
        "tensorflow.compat.v1.linalg",
        "tensorflow.compat.v1.math",
        "tensorflow.compat.v2",
        "tensorflow.compat.v2.nn",
        "tensorflow.compat.v2.linalg",
        "tensorflow.compat.v2.math",
        "tensorflow.dtypes",
        "tensorflow.experimental.numpy",
        "tensorflow.keras.activations",
        "tensorflow.keras.backend",
        "tensorflow.keras.metrics",
        "tensorflow.keras.layers",
        "tensorflow.keras.losses",
        "tensorflow",
        "tensorflow.__operators__",
        "tensorflow.linalg",
        "tensorflow.random",
        "tensorflow.nn",
        "tensorflow.math",
        "tensorflow.signal",
        "tensorflow.image",
    ],
    "torch": [
        "torch",
        "torchvision.ops",
        "torch.nn.functional",
        "torch.fft",
        "torch.linalg",
        "torch.signal",
        "torch.special",
        "torch.nn.utils.rnn",
    ],
    "ivy": ["ivy"],
}

CLASSES_TO_WRAP = {
    "numpy": [("numpy", "ndarray")],
    "paddle": [("paddle", "Tensor")],
    "jax": [
        ("jaxlib.xla_extension", "DeviceArray"),
        ("jaxlib.xla_extension", "ArrayImpl"),
    ],
    "tensorflow": [
        ("tensorflow._api.v2.__internal__", "EagerTensor"),
        ("tensorflow", "Tensor"),
        ("tensorflow.python.ops.resource_variable_ops", "ResourceVariable"),
        ("tensorflow", "Variable"),
    ],
    "torch": [("torch", "Tensor")],
    "ivy": [("ivy", "Array")],
}


def PRIVATE_CLASSES_TO_WRAP(fw):
    return {
        "numpy": [],
        "paddle": [],
        "jax": version_resolver(
            get_version(fw),
            {
                "0.4.7 and above": [
                    ("jax._src.numpy.array_methods", "_IndexUpdateRef"),
                    ("jax._src.numpy.array_methods", "_IndexUpdateHelper"),
                ],
                "0.4.6 and below": [
                    ("jax._src.numpy.lax_numpy", "_IndexUpdateRef"),
                    ("jax._src.numpy.lax_numpy", "_IndexUpdateHelper"),
                ],
            },
        ),
        "tensorflow": [],
        "torch": [],
        "ivy": [],
    }[fw]


FUNCTIONS_ATTRS_NOT_TO_WRAP = {
    "numpy": ["format_float_positional", "dragon4_positional"],
    "paddle": [],
    "jax": [
        "pjit",
        "_single_device_array_to_np_array",
        "__array__",
        "get_backend",
        "tree_flatten",
        "tree_unflatten",
        "canonicalize_platform",
        "backends",
        "devices",
        "device",
        "device_buffer",
        "platform",
        "clone",
        "block_host_until_ready",
        "block_until_ready",
        "copy_to_device",
        "copy_to_host_async",
        "_copy_single_device_array_to_host_async",
        "copy_to_remote_device",
        "delete",
        "is_deleted",
        "is_known_ready",
        "is_ready",
        "on_device_size_in_bytes",
        "to_py",
        "unsafe_buffer_pointer",
        "xla_dynamic_shape",
        "xla_shape",
        "default_prng_impl",
        "flattened_fun_in_tree",
        "flatten_fun",
        "flatten_fun_nokwargs",
        "get_aval",
        "concrete_aval",
        "function transformation_with_aux",
        "flatten_fun_for_vmap",
        "replace_thread_exc_traceback",
        "path_starts_with",
        "include_frame",
        "ignore_known_hidden_frame",
        "add_call_stack_frames",
        "format_exception_only",
        "xla_callable",
        "tree_leaves",
        "tree_map",
    ],
    "tensorflow": [
        "as_dtype",
        "flatten",
        "pack_sequence_as",
        "map_structure",
        "deprecated_argument_lookup",
        "_variable_call",
        "getitem",
    ],
    "torch": [
        "__torch_function__",
        "unpack_dual",
        "classes",
        "torch",
        "is_grad_enabled",
        "get_default_dtype",
        "cpu",
        "set_",
        "requires_grad_",
        "load",
    ],
    "ivy": [
        "__init__",
        "args_to_ivy",
        "variable",
        "nested_map",
        "map_nest_at_index",
        "set_nest_at_index",
        "set_nest_at_indices",
        "multi_index_nest",
        "index_nest",
        "to_ivy",
        "exists",
        "default",
        "container_types",
        "to_native",
        "nested_argwhere",
        "map_nest_at_indices",
        "is_native_array",
        "current_backend_str",
        "is_array",
        "is_variable",
        "current_backend",
        "is_ivy_array",
        "get_backend",
        "with_grads",
        "check_elem_in_list",
        "check_isinstance",
        "check_all",
        "args_to_native",
        "nested_any",
        "is_ivy_container",
        "check_true",
        "handle_exceptions",
        "to_list",
        "as_ivy_dev",
        "dev",
        "dtype",
        "promote_types_of_inputs",
        "default_device",
        "handle_nestable",
        "outputs_to_ivy_arrays",
        "handle_array_like",
        "inputs_to_native_arrays",
        "inputs_to_ivy_arrays",
        "handle_out_argument",
        "as_int_dtype",
        "as_ivy_dtype",
        "gpu_is_available",
        "default_float_dtype",
        "is_float_dtype",
        "set_backend",
        "previous_backend",
        "del_global_attr",
        "check_false",
        "infer_device",
        "integer_arrays_to_float",
        "infer_dtype",
        "to_numpy",
        "as_native_dev",
        "is_int_dtype",
        "as_native_dtype",
        "default_dtype",
        "set_global_attr",
        "set_backend_to_specific_version",
    ],
}

# Union of builtin array methods of native framework arrays that need to be wrapped
ARRAY_BUILTINS = [
    "_rewriting_take",
    "_slice_helper",
    "_threshold",
    "__neg__",
    "__pow__",
    "__rpow__",
    "__add__",
    "__radd__",
    "__iadd__",
    "__sub__",
    "__rsub__",
    "__isub__",
    "__mul__",
    "__mod__",
    "__rmod__",
    "__rmul__",
    "__imul__",
    "__matmul__",
    "__rmatmul__",
    "__truediv__",
    "__rtruediv__",
    "__itruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__ifloordiv__",
    "__idiv__",
    "__abs__",
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    "__and__",
    "__rand__",
    "__or__",
    "__ror__",
    "__invert__",
    "__xor__",
    "__rxor__",
    "__getitem__",
    "__setitem__",
    "__getattr__",
    "__setattr__",
    "__getattribute__",
    "__init__",
    "__repr__",
]


# A list of private functions that need to be wrapped
PRIVATE_FUNCTIONS_TO_WRAP = [
    "_pad",  # temp patch for ODSC kornia demo
    "_add_dispatch",  # need to wrap tensorflow private methods
]


# Special Cases #
# ------------- #

# Attributes for which the __getattribute__ (or similar) call should
# be included in the graph for each framework
GRAPH_ATTRIBUTES = {
    "numpy": ["shape", "ndim", "size", "itemsize", "T", "strides"],
    "paddle": ["shape", "strides"],
    "jax": ["at", "shape", "strides"],
    "tensorflow": ["shape", "strides"],
    "torch": ["data", "requires_grad", "shape", "T", "H", "mT", "mH", "strides"],
    "ivy": ["shape", "strides"],
}

# Framework functions which operate inplace and give no return
INPLACE_FUNCTIONS_WITHOUT_RET = {
    "numpy": ["copyto", "put_along_axis"],
    "paddle": [],
    "jax": [],
    "tensorflow": [],
    "torch": [],
    "ivy": [],
}

# Framework methods which operate inplace and give no return
# eg. x = np_array.__setitem__(0, 1)
# gives x = None (as there is no return), but np_array will still be updated
INPLACE_METHODS_WITHOUT_RET = {
    "numpy": [
        "__init__",
        "__setitem__",
        "resize",
        "sort",
        "partition",
        "fill",
        "setflags",
        "itemset",
    ],
    "paddle": ["__init__", "__setitem__"],
    "jax": ["__init__"],
    "tensorflow": ["__init__", "assign", "assign_sub", "assign_add"],
    "torch": ["__init__", "__setitem__"],
    "ivy": ["__init__", "__setitem__"],
}

# Functions which generate random arrays/tensors in each framework
GENERATOR_FUNCTIONS = {
    "numpy": [
        "uniform",
        "normal",
        "rand",
        "randn",
        "random",
        "randint",
        "random_integers",
        "random_sample",
        "beta",
        "binomial",
        "chisquare",
        "dirichlet",
        "exponential",
        "f",
        "gamma",
        "geometric",
        "gumbel",
        "hypergeometric",
        "laplace",
        "logistic",
        "lognormal",
        "logseries",
        "multinomial",
        "multivariate_normal",
        "negative_binomial",
        "noncentral_chisquare",
        "noncentral_f",
        "pareto",
        "poisson",
        "rayleigh",
        "standard_cauchy",
        "standard_exponential",
        "standard_gamma",
        "standard_normal",
        "standard_t",
        "trinagular",
        "vonmises",
        "wald",
        "weibull",
        "zipf",
    ],
    "paddle": [
        "bernoulli",
        "multinomial",
        "normal",
        "poisson",
        "rand",
        "randint",
        "randint_like",
        "randn",
        "randperm",
        "standard_normal",
        "uniform",
    ],
    "jax": [
        "ball",
        "bernoulli",
        "beta",
        "categorical",
        "cauchy",
        "dirichlet",
        "double_sided_maxwell",
        "exponential",
        "gamma",
        "generalized_normal",
        "gumbel",
        "laplace",
        "loggamma",
        "logistic",
        "maxwell",
        "multivariate_normal",
        "normal",
        "orthogonal",
        "pareto",
        "poisson",
        "rademacher",
        "randint",
        "t",
        "truncated_normal",
        "uniform",
        "weibull_min",
    ],
    "tensorflow": [
        "random_uniform",
        "random_normal",
        "categorical",
        "random_gamma",
        "truncated_normal",
        "random_poisson_v2",
    ],
    "torch": [
        "rand",
        "normal",
        "multinomial",
        "randint",
        "bernoulli",
        "poisson",
        "randn",
        "randperm",
    ],
    "ivy": ["random_uniform", "random_normal", "multinomial", "randint"],
}

# Special cases- to avoid needing to create weird things in the frontends
NATIVE_TO_FRONTEND_PATH = {
    "tensorflow._api.v2.__internal__.EagerTensor": "tensorflow.Tensor",
    "jaxlib.xla_extension.ArrayImpl": "jax.Array",
}

# Special cases in source gen where need to add special imports to make
# some absolute function calls work e.g. `tensorflow.python.ops`
FN_PATH_TO_IMPORT_PATH= {
    "tensorflow.python.ops.resource_variable_ops": "from tensorflow.python.ops.resource_variable_ops import <placeholder>",
}

# Cases where a function call beginning with any of these strings can instead be converted to
# a method call in the generated source code for improved robustness
# for example: numpy.ndarray.astype(x, c4350497972) -> x.astype(c4350497972)
FUNCTION_TO_METHOD_CONVERSIONS = [
    "numpy.ndarray",
    "torch.Tensor",
]

# Framework specific classes that we need to track by converting to a
# subclass of TrackedVarProxy (eg. TrackedTensorShapeProxy)
CLASSES_TO_TRACK = {
    "numpy": [],
    "paddle": [],
    "jax": [],
    "tensorflow": ["TensorShape"],
    "torch": ["Size"],
    "ivy": ["Shape"],
}

# Functions which return an arbitrary variable which needs to be tracked (int, tuple, etc)
# (except methods of TrackedVarProxy, which already tracks return variables)
FNS_TO_TRACK = {
    "numpy": [
        "item",
        "shape",
        "tolist",
    ],
    "paddle": [
        "item",
        "tolist",
    ],
    "jax": [
        "broadcast_shapes",
        "item",
        "shape",
        "tolist",
    ],
    "tensorflow": [
        "item",
        "tolist",
    ],
    "torch": [
        "item",
        "numel",
        "size",
        "stride",
        "tolist",
    ],
    "ivy": ["tolist"],
}

# Python's builtin functions that we should track
# eg. so we can track things like `x = min(array)`
# which are not framework specific operations
BUILTIN_FNS_TO_TRACK = [
    "min",
    "max",
    "sum",
]

# Python's builtin methods that we should track
# e.g. [].append or ''.join
BUILTIN_METHODS_TO_TRACK = [
    "join",
]

# Python's builtin type castings we should track
# eg. we often need to track through builtin type castings
# like `z = int(x / y)` to get a complete graph
BUILTIN_TYPES_TO_TRACK = [
    int,
    float,
    str,
    tuple,
    list,
    dict,
    bytes,
]

BUILTIN_CALLABLES_TO_TRACK = [len, *BUILTIN_TYPES_TO_TRACK]

# Attributes which return an arbitrary variable that needs to be
# tracked via our TrackedVarProxy (int, tuple, etc)
ATTRS_TO_TRACK = {
    "numpy": ["shape", "ndim", "size", "itemsize", "strides"],
    "paddle": ["shape", "strides"],
    "jax": ["shape", "strides"],
    "tensorflow": ["strides"],
    "torch": ["strides"],
    "ivy": ["strides"],
}

# Functions part of the standard libs (eg: math, itertools etc.)
BUILTIN_MODULES_TO_TRACK = ["math", ]
# Functions which return a list of tensors/arrays which need to be tracked
TENSOR_LIST_FNS_TO_TRACK = {
    "numpy": ["tolist"],
    "paddle": [],
    "jax": [],
    "tensorflow": ["unstack"],
    "torch": [],
    "ivy": [],
}

CLASS_ATTRS_NOT_TO_TRACK = {
    "numpy": {},
    "paddle": {},
    "jax": {
        "Array": [
            "device_buffer",
            "is_fully_replicated",
        ],
        "ArrayImpl": [
            "device_buffer",
            "is_fully_replicated",
        ],
    },
    "tensorflow": {},
    "torch": {},
    "ivy": {},
}

# kwargs to the forward pass of a module which we consider
# to be setting the training mode of that module pass
# (best to keep this is order of most common -> least common)
TRAIN_KWARGS = [
    "training",
    "train",
    "is_training",
]

# Higher order functions for which we will need to trace callback subgraph(s)
HIGHER_ORDER_FNS_TO_TRACE = {
    "jax": [
        "while_loop",
    ],
    "numpy": [],
    "paddle": [],
    "tensorflow": [
        "rnn",
        "while_loop",
        "while_loop_v2",
        # "cond_for_tf_v2",
    ],
    "torch": [],
    "ivy": [
        "while_loop",
    ],
}

# Functions that can be ignored when searching for possible truncation subgraphs
TRUNCATION_FNS_TO_IGNORE = {
    "jax": [],
    "numpy": [],
    "paddle": [],
    "tensorflow": [
        "convert_to_tensor_v2_with_dispatch",
        "__getattribute__",
    ],
    "torch": [],
}

# check whether we need to apply an optimization to remove
# unnecessary transpose calls due to data_format differences
# when transpiling from torch (channel first) to tensorflow (channel last)
TRANSPOSE_OPTIMIZATION_FUNCTIONS = [
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "max_unpool1d",
    "max_unpool2d",
    "max_unpool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "batch_norm",
    "group_norm",
    "instance_norm",
    "layer_norm",
    "local_response_norm",
]
