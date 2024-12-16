# Graph Tracer

A powerful tool for generating optimized computational graphs from arbitrary framework-agnostic code.

## Overview

The Graph Tracer transforms arbitrary code (either Ivy or native framework code like TensorFlow, PyTorch, JAX) into an efficient Directed Acyclic Graph (DAG) that represents the data flow. This computational graph optimizes execution by:
- Removing redundant operations
- Stripping unused functions
- Preserving only the essential path from inputs to outputs

## Key Features

- **Framework Agnostic**: Works with both Ivy and native framework code
- **Optimization**: Removes wrapping logic and unused functions
- **Performance**: Significantly improves execution efficiency

## Directory Structure

```
tracer/
├── special_ops/                # Handling tracing of higher-order functions
│   ├── builtin_helpers.py     # Helpers for built-in(len, min, max etc.) function tracing
│   ├── vmap_helpers.py        # Helpers for vectorized map(vmap) operations
├── conversion.py              # Handling conversions between frameworks/proxies
├── exchange.py                # Graph serialization/deserialization utilities
├── globals.py                 # Global variables for tracing process
├── graph.py                   # Graph construction and manipulation
├── helpers.py                 # General utility functions for tracing
├── param.py                   # Parameter handling and management
├── reloader.py                # Module reloading utilities
├── tracked_var_proxy.py       # Proxy classes for variable tracking
├── tracked_var_replacements.py # Replacement functions for tracked variables
├── tracer.py                  # Core tracing functionality
└── wrapping.py                # Function wrapping utilities for logging
```

## How It Works

The compilation process consists of four main stages:

### 1. [Setup](https://github.com/ivy-llc/tracer-transpiler/blob/main/tracer/tracer.py#L719)
- Import the underlying framework
- Set up necessary environment (creating a graph instance, module reloading, etc.)

### 2. [Function Wrapping and Tracing](https://github.com/ivy-llc/tracer-transpiler/blob/main/tracer/wrapping.py#L285)

The wrapping stage enables dynamic tracking of primitive function calls during execution:

```python
def _wrap_function_for_op_logging(fn: Callable, graph: Graph, ...) -> Callable:
    def _tracing_function(*args, **kwargs):
        # Log function call information
        # Store in graph
        # Execute original function
        return result
    return _tracing_function
```

**Key Components:**
- **Wrapper Creation**: Creates logging wrappers for primitive functions
- **Function Binding**: Rebinds function names to wrappers in module namespaces
- **Information Logging**: Tracks:
  - Memory addresses (ids) of inputs/outputs
  - Function call sequences
  - Argument relationships

Example of wrapper binding (aka: monkey patching):
```python
# Original function
torch.nn.functional.linear(x, weight, bias)

# After wrapping
torch.nn.functional.linear = _wrap_function_for_op_logging(torch.nn.functional.linear, graph)
torch.nn.functional.linear(x, weight, bias) # This will now be traced and stored in graph
```

### 3. [Graph Construction](https://github.com/ivy-llc/tracer-transpiler/blob/main/tracer/graph.py#L961)

The graph is constructed through a backward traversal process:

```python
def connect():
    """Connects functions together into the final graph."""
    # Start from output
    # Traverse backwards through operations
    # Track parameters and functions
    return optimized_graph
```

**Construction Process:**
1. **Output Registration**: 
   - Start from function output
   - Record output information
   - Initialize graph structure

2. **Backward Traversal**:
   - Compare output IDs with node outputs
   - Link outputs to producing nodes
   - Track parameters in `_tmp_sub_param_dict`
   - Record functions in `_tmp_sub_functions`

3. **Node Connection**:
   - Link nodes based on input/output relationships
   - Build optimal execution path
   - Remove unused nodes

Example of graph construction:
```python
def example_function(x):
    y = ivy.mean(x)     # Node 1
    z = ivy.sqrt(y)     # Node 2
    return z

# Graph construction starts from z, links to sqrt node,
# then to mean node, and finally to input x
```

**Performance Comparison:**
```python
# Original vs Compiled Performance
x = ivy.array([1., 2., 3.])

# Original: ~0.57ms
start = time.time()
result = fn(x)
print(f"Original: {time.time() - start:.4f}s")

# Compiled: ~0.22ms
start = time.time()
result = compiled_fn(x)
print(f"Compiled: {time.time() - start:.4f}s")
```

###  Pros/Cons of Graph Compilation

1. **Pros**:
   - Removes unused operations and python overhead enabling native compiler optimizations (ie: tf.function, jax.jit)

2. **Cons**:
   - Cannot fully capture python dynamsim (control flow, dynamic shapes etc.)

## Usage Example

```python
import ivy
from tracer.tracer import trace_graph
ivy.set_backend("torch")

# Define a function with multiple operations
def fn(x):
    y = ivy.sum(x)
    z = ivy.prod(x)
    a = ivy.sin(y)
    b = ivy.cos(z)
    c = ivy.tan(z)
    i = ivy.round(a)
    j = ivy.floor(b)
    k = ivy.ceil(c)
    return i, j, k

# Compile the function
x = ivy.array([1.])
comp_fn = trace_graph(fn, args=(x,), to="torch")

# Compare performance
import time

# Original function
start = time.time()
fn(x)
print(f"Original: {time.time() - start:.4f}s")  # ~0.4957s

# Compiled function
start = time.time()
comp_fn(x)
print(f"Compiled: {time.time() - start:.4f}s")  # ~0.0006s
```

## Contributing

Contributions are welcome! Please read our contribution guidelines before submitting pull requests.

## License

This project is licensed under the Apache License 2.0.

