import json
from typing import Dict
from ivy.transpiler.caching.assets.torch_cache_assets import (
    ALL_TORCH_FRONTEND_FUNCTIONS,
    ALL_TORCH_LAYERS,
)
from ivy.transpiler.caching.assets.ivy_cache_assets import (
    ALL_IVY_FUNCTIONS,
)


def generate_matrix(data: Dict, source: str, target: str):
    return [
        {"func_name": func_name, "module": module, "source": source, "target": target}
        for module, funcs in data.items()
        for func_name in funcs
    ]


def main(matrix_type):
    if matrix_type == "torch_frontend_functions_matrix":
        matrix = generate_matrix(
            data=ALL_TORCH_FRONTEND_FUNCTIONS,
            source="torch_frontend",
            target="tensorflow",
        )
    elif matrix_type == "torch_layers_matrix":
        matrix = generate_matrix(
            data=ALL_TORCH_LAYERS, source="torch", target="tensorflow"
        )
    elif matrix_type == "ivy_functions_matrix":
        matrix = generate_matrix(
            data=ALL_IVY_FUNCTIONS, source="ivy", target="tensorflow"
        )
    else:
        raise ValueError("Invalid matrix type")

    # Convert to JSON string and print it
    print(json.dumps(matrix))


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
