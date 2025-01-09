# Transformations

This directory contains the AST transformers used in the source-to-source translation process.

## Structure

The transformers follow a modular design where each type of transformation is handled by a dedicated transformer module:

```
transformers/
├── annotation_transformer      # Handles type annotations
├── canonicalize_transformer   # Canonicalizes imports and function calls to use full module paths
├── closure_transformer       # Manages closure scope variables
├── decorator_transformer    # Processes function decorators
├── deletion_transformer    # Removes specific AST nodes/code segments
├── docstring_transformer  # Modifies and transforms docstrings
├── dunders_transformer   # Handles special methods (__getitem__, __setitem__, etc.)
├── globals_transformer  # Manages global variable transformations
├── inject_transformer   # Injects new code/functionality
├── method_transformer  # Handles class method transformations
├── native_layers_transformer  # Maps and replaces neural network layers (nn.Linear, nn.Conv2d, etc.)
├── postprocessing_transformer # Performs final cleanup and framework-specific adjustments
├── preprocessing_transformer  # Handles initial AST modifications
├── recursive_transformer     # Manages nested function calls and recursive transformations
├── rename_transformer       # Handles identifier renaming (variables, functions, classes)
└── typing_transformer     # Processes Python type hints
```

## Design Pattern

Each transformer follows a consistent design pattern:

1. **Base Transformer**: Each transformer module contains a `base_transformer.py` that defines the base class with core functionality.

2. **Stage-Specific Implementations**: The base transformer is extended for each stage of the transpilation process. For example:
   ```python
   recursive_transformer/
   ├── base_transformer.py              # Defines BaseRecurser
   ├── frontend_torch_recursive_transformer.py  # For torch_frontend-to-ivy stage
   └── ivy_recursive_transformer.py            # For ivy-to-target stage
   ```

3. **Configuration**: Each transformer has associated metadata stored in `transformations/configurations/`. For example:
   ```python
   configurations/
   ├── base_transformer_config.py
   ├── ivy_postprocessing_transformer_config.py  # Contains dtype mappings, etc.
   └── frontend_torch_postprocessing_transformer_config.py
   ```

## Example Transformation Flow

Here's how a typical transformation might work:

1. **Preprocessing**: Initial AST modifications 
2. **Core Transformations**: Multiple passes of specific transformers
3. **Postprocessing**: Final cleanup and framework-specific adjustments

For example, converting a PyTorch function to Ivy's Frontend IR:

```python
# Original PyTorch code
import torch
import torch.nn as nn
def func(x: torch.Tensor) -> torch.Tensor:
    return nn.functional.linear(10, 5)(x) + x.mean()
```

```diff
# After canonicalization transformer
def func(x: torch.Tensor) -> torch.Tensor:
-     return nn.functional.linear(10, 5)(x) + x.mean()
+     return torch.nn.functional.linear(x, weight, bias) + x.mean()
```

```diff
# After type transformation
- def func(x: torch.Tensor) -> torch.Tensor:
+ def func(x: typing.Any):
      return torch.nn.functional.linear(x, weight, bias) + x.mean()
```

```diff
# After method transformation
def func(x: typing.Any):
-     return torch.nn.functional.linear(x, weight, bias) + x.mean()
+     return torch.nn.functional.linear(x, weight, bias) + torch.Tensor.mean(x)
```

```...```
```diff
# Final Code Generation
+ import ivy.functional.frontends.torch as torch
- def func(x: typing.Any):
+ def Translated_func(x: typing.Any):
    return torch.nn.functional.linear(x, weight, bias) + torch.Tensor.mean(x)
```

Each transformation pass focuses on a specific aspect of the code:
- **Canonicalization**: Standardizes function calls and imports
- **Type Transformer**: Handles type annotations 
- **Method Transformer**: Converts method calls to function calls
- **Code Generation**: Final cleanup and import adjustments

## Adding New Transformers

When adding a new transformer:
1. Create a new directory under `transformers/`
2. Implement the base transformer class in `base_transformer.py`
3. Add stage-specific implementations as needed
4. Create corresponding configuration files in `configurations/`
