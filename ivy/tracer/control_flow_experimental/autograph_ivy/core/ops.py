# Do not change directly.
import threading
import collections


GRAPH_MODE = 0
EAGER_MODE = 1
SYNC = 0
ASYNC = 1

default_execution_mode = EAGER_MODE


ContextSwitch = collections.namedtuple(
    "ContextSwitch", ["is_building_function", "enter_context_fn", "device_stack"]
)


class _AtomicCounter(object):
    """A simple atomic counter."""

    __slots__ = ["_value", "_lock"]

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment_and_get(self):
        with self._lock:
            self._value += 1
            return self._value


def eager_mode():
    """Context-manager to enable eager execution for the current thread."""
    return context()._mode(EAGER_MODE)  # pylint: disable=protected-access


def scope_name():
    """Name of the current scope."""
    return context().scope_name


# `_ContextSwitchStack` is a `threading.local` to match the semantics of
# ``DefaultGraphStack`, which is also a `threading.local`.
class _ContextSwitchStack(threading.local):
    """A thread-local stack of context switches."""

    def __init__(self, eager):
        super().__init__()
        self.stack = []
        if eager:
            # Initialize the stack with a pointer to enter the eager context; this
            # ensures that the fact that eager execution was enabled is propagated
            # across threads, since (1) `enable_eager_execution` modifies a
            # process-level flag (`default_execution_mode`) and (2) `__init__` is
            # called each time a threading.local object is used in a separate thread.
            self.push(
                is_building_function=False,
                enter_context_fn=eager_mode,
                device_stack=None,
            )

    def push(self, is_building_function, enter_context_fn, device_stack):
        """Push metadata about a context switch onto the stack.

        A context switch can take any one of the two forms: installing a graph as
        the default graph, or entering the eager context. For each context switch,
        we record whether or not the entered context is building a function.

        Args:
          is_building_function: (bool.) Whether the context is building a function.
          enter_context_fn: (function.) A callable that executes the context switch.
            For example, `graph.as_default` or `eager_mode`.
          device_stack: If applicable, the device function stack for this graph.
            When breaking out of graphs in init_scope, the innermost nonempty device
            stack is used. Eager contexts put `None` here and the value is never
            used.
        """

        self.stack.append(
            ContextSwitch(is_building_function, enter_context_fn, device_stack)
        )

    def pop(self):
        """Pop the stack."""

        self.stack.pop()


_context = None
_context_lock = threading.Lock()
_context_id_counter = _AtomicCounter()


def _set_context_locked(ctx):
    global _context
    ctx.mark_as_global_context()
    _context = ctx


def _create_context():
    with _context_lock:
        if _context is None:
            ctx = Context()
            _set_context_locked(ctx)


def context():
    """Returns a singleton context object."""
    if _context is None:
        _create_context()
    return _context


class EagerContextThreadLocalData:
    @property
    def scope_name(self):
        """Returns scope name for the current thread."""
        return self._scope_name

    @scope_name.setter
    def scope_name(self, s):
        """Sets scope name for the current thread."""
        self._scope_name = s


class name_scope_v2(object):
    """A context manager for use when defining a Python op.

    This context manager pushes a name scope, which will make the name of all
    operations added within it have a prefix.

    For example, to define a new Python op called `my_op`:

    ```python
    def my_op(a, b, c, name=None):
      with tf.name_scope("MyOp") as scope:
        a = tf.convert_to_tensor(a, name="a")
        b = tf.convert_to_tensor(b, name="b")
        c = tf.convert_to_tensor(c, name="c")
        # Define some computation that uses `a`, `b`, and `c`.
        return foo_op(..., name=scope)
    ```

    When executed, the Tensors `a`, `b`, `c`, will have names `MyOp/a`, `MyOp/b`,
    and `MyOp/c`.

    Inside a `tf.function`, if the scope name already exists, the name will be
    made unique by appending `_n`. For example, calling `my_op` the second time
    will generate `MyOp_1/a`, etc.
    """

    __slots__ = ["_name", "_exit_fns"]

    def __init__(self, name):
        """Initialize the context manager.

        Args:
          name: The prefix to use on all names created within the name scope.

        Raises:
          ValueError: If name is not a string.
        """
        if not isinstance(name, str):
            raise ValueError("name for name_scope must be a string.")
        self._name = name
        self._exit_fns = []

    @property
    def name(self):
        return self._name

    def __enter__(self):
        """Start the scope block.

        Returns:
          The scope name.
        """
        ctx = context()
        # Names are not auto-incremented in eager mode.
        # A trailing slash breaks out of nested name scopes, indicating a
        # fully specified scope name, for compatibility with Graph.name_scope.
        # This also prevents auto-incrementing.
        old_name = None
        name = self._name
        if not name:
            scope_name = ""
        elif name[-1] == "/":
            scope_name = name
        elif old_name:
            scope_name = old_name + name + "/"
        else:
            scope_name = name + "/"
        ctx.scope_name = scope_name

        def _restore_name_scope(*_):
            ctx.scope_name = old_name

        self._exit_fns.append(_restore_name_scope)

        return scope_name

    def __exit__(self, type_arg, value_arg, traceback_arg):
        self._exit_fns.pop()(type_arg, value_arg, traceback_arg)
        return False  # False values do not suppress exceptions

    def __getstate__(self):
        return self._name, self._exit_fns

    def __setstate__(self, state):
        self._name = state[0]
        self._exit_fns = state[1]


class Context:
    """Environment in which eager operations execute."""

    # TODO(agarwal): create and link in some documentation for `execution_mode`.
    # pylint: disable=redefined-outer-name
    def __init__(
        self, config=None, device_policy=None, execution_mode=None, server_def=None
    ):
        """Creates a new Context.

        Args:
          config: (Optional.) A `ConfigProto` protocol buffer with configuration
            options for the Context. Note that a lot of these options may be
            currently unimplemented or irrelevant when eager execution is enabled.
          device_policy: (Optional.) What policy to use when trying to run an
            operation on a device with inputs which are not on that device. When set
            to None, an appropriate value will be picked automatically. The value
            picked may change between TensorFlow releases.  Defaults to
            DEVICE_PLACEMENT_SILENT.
            Valid values:
            - DEVICE_PLACEMENT_EXPLICIT: raises an error if the placement is not
              correct.
            - DEVICE_PLACEMENT_WARN: copies the tensors which are not on the right
              device but raises a warning.
            - DEVICE_PLACEMENT_SILENT: silently copies the tensors. This might hide
              performance problems.
            - DEVICE_PLACEMENT_SILENT_FOR_INT32: silently copies int32 tensors,
              raising errors on the other ones.
          execution_mode: (Optional.) Policy controlling how operations dispatched
            are actually executed. When set to None, an appropriate value will be
            picked automatically. The value picked may change between TensorFlow
            releases.
            Valid values:
            - SYNC: executes each operation synchronously.
            - ASYNC: executes each operation asynchronously. These operations may
              return "non-ready" handles.
          server_def: (Optional.) A tensorflow::ServerDef proto. Enables execution
            on remote devices. GrpcServers need to be started by creating an
            identical server_def to this, and setting the appropriate task_indexes,
            so that the servers can communicate. It will then be possible to execute
            operations on remote devices.

        Raises:
         ValueError: If execution_mode is not valid.
        """
        # This _id is used only to index the tensor caches.
        # TODO(iga): Remove this when tensor caches are moved to C++.
        self._id = _context_id_counter.increment_and_get()
        self._config = config
        self._thread_local_data = EagerContextThreadLocalData()
        self._context_switches = _ContextSwitchStack(True)
        self._initialize_lock = threading.Lock()
        self._initialized = False
        if execution_mode is None:
            execution_mode = SYNC
        self._default_is_async = execution_mode == ASYNC

        self._is_global_context = False

    @property
    def context_switches(self):
        """Returns a stack of context switches."""
        return self._context_switches

    @property
    def scope_name(self):
        """Returns scope name for the current thread."""
        return self._thread_local_data.scope_name

    @scope_name.setter
    def scope_name(self, s):
        """Sets scope name for the current thread."""
        self._thread_local_data.scope_name = s

    def mark_as_global_context(self):
        self._is_global_context = True


def name_scope(name, default_name=None, values=None, skip_on_eager=True):
    """Internal-only entry point for `name_scope*`.

    Internal ops do not use the public API and instead rely on
    `ops.name_scope` regardless of the execution mode. This function
    dispatches to the correct `name_scope*` implementation based on
    the arguments provided and the current mode. Specifically,

    * if `values` contains a graph tensor `Graph.name_scope` is used;
    * `name_scope_v1` is used in graph mode;
    * `name_scope_v2` -- in eager mode.

    Args:
      name: The name argument that is passed to the op function.
      default_name: The default name to use if the `name` argument is `None`.
      values: The list of `Tensor` arguments that are passed to the op function.
      skip_on_eager: Indicates to return NullContextmanager if executing eagerly.
        By default this is True since naming tensors and operations in eager mode
        have little use and cause unnecessary performance overhead. However, it is
        important to preserve variable names since they are often useful for
        debugging and saved models.

    Returns:
      `name_scope*` context manager.
    """
    name = default_name if name is None else name
    return name_scope_v2(name or "")
