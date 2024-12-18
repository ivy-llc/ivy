import json
import os
import importlib
import sys
import pytest
import logging

from ivy.transpiler import transpile
from ivy.transpiler.utils.api_utils import get_function_from_modules


def populate_cache(func_name: str, module: str, source: str, target: str):
    logging.debug(f"Populating cache for {func_name} in {source}")
    os.environ["UPDATE_S2S_CACHE"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"

    import ivy

    ivy.set_backend(target)

    if "frontend" in source:
        module = importlib.import_module(f"ivy.functional.frontends.{module}")
    else:
        module = importlib.import_module(module)

    # Get function dynamically from the appropriate module
    func = get_function_from_modules(func_name, [module])
    transpile(func, source=source, target=target)


# Define a pytest function to handle each item in the batch
@pytest.mark.parametrize("item", json.loads(sys.argv[1]))
def test_populate_cache(item):
    func_name = item["func_name"]
    module = item["module"]
    source = item["source"]
    target = item["target"]

    try:
        # Call the populate_cache function for each function/layer in the batch
        populate_cache(func_name, module, source, target)
    except Exception as e:
        pytest.fail(f"Failed to populate cache for {func_name} in {module}: {e}")


if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__])
