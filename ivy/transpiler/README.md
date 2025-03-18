# Source to Source Translator

A powerful Python source-to-source translation tool that converts code between deep learning frameworks using Ivy as an intermediate representation (IR).

## Overview

The translator performs framework conversions through a three-stage pipeline:

1. **Source Framework → Frontend IR**: Converts source framework code to Ivy's frontend representation
2. **Frontend IR → Ivy Core**: Transforms frontend IR to Ivy's core representation 
3. **Ivy Core → Target Framework**: Converts Ivy code to target framework

Each stage involves multiple AST transformation passes that systematically modify the code while preserving semantics.

## Key Features

- **Multi-Framework Support**: Convert any PyTorch code to JAX, TensorFlow, and NumPy
- **AST-Level Transformations**: Precise code modifications through Python's Abstract Syntax Tree
- **Modular Design**: Composable transformers for different aspects of code conversion. Easily extendable to support new frameworks
- **Configurable Pipeline**: Customizable transformation passes for each stage

## Usage

```python
import ivy
import torch

def torch_model(x):
    return torch.nn.functional.linear(10, 5)(x)

# Convert PyTorch model to TensorFlow
tf_model = ivy.transpile(
    torch_model,
    source="torch", 
    target="tensorflow",
    output_dir="transpiled_models"
)
```

## Directory Structure

```
source_to_source_translator/
├── main.py                     # Main entry point exposing transpile() API
├── translations/               # Core translation infrastructure
│   ├── translator.py          # Main Translator class implementation
│   └── data/                  # Data classes for objects and globals
├── transformations/           # AST transformation modules
│   ├── transformers/          # Individual transformer implementations
│   │   ├── annotation_transformer/     # Type annotation handling
│   │   ├── canonicalize_transformer/   # Import and call canonicalization
│   │   ├── method_transformer/         # Method call conversions
│   │   └── ...                        # Other transformers
│   └── configurations/        # Transformer-specific configs
├── configs/                   # Stage-specific translation configs
├── utils/                     # Utility functions
│   ├── ast_utils.py          # AST manipulation and source code generation
│   ├── cache_utils.py        # Caching utilities
│   └── logging_utils.py      # Logging infrastructure
│   └── ...                   # other utilities
└── exceptions/               # Custom exception classes
```

## Translation Process

1. [**Initialization** ](https://github.com/ivy-llc/tracer-transpiler/blob/main/source_to_source_translator/main.py#L767)
   - Parse source code to AST
   - Load appropriate stage configurations
   - Initialize transformer pipeline

2. [**Transformation Pipeline**](./transformations/README.md)
   - **Preprocessing**: Initial AST cleanup and normalization
   - **Core Transformations**: Multiple passes including:
     - Type annotation handling
     - Import canonicalization
     - Method call conversions
     - Closure variable management
     - Special method (dunder) handling
     - ... # and more
   - **Postprocessing**: Framework-specific adjustments

3. [**Code Generation**](https://github.com/ivy-llc/tracer-transpiler/blob/main/source_to_source_translator/utils/ast_utils.py#L2775)
   - Convert transformed AST back to source code
   - Apply formatting and cleanup
   - Save to specified output directory

## Output Structure

```
ivy_transpiled_outputs/
├── tensorflow_outputs/        # Target: TensorFlow
│   ├── __init__.py
│   └── model.py
├── torch_frontend_outputs/    # Target: PyTorch Frontend
│   ├── __init__.py
│   └── model.py
└── ivy_outputs/              # Target: Ivy
    ├── __init__.py
    └── model.py
```


## Contributing

Contributions are welcome! Please read our contribution guidelines before submitting pull requests.

## License

This project is licensed under the Apache License 2.0.

