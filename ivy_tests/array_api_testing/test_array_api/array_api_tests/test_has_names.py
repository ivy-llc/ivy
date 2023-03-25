"""
This is a very basic test to see what names are defined in a library. It
does not even require functioning hypothesis array_api support.
"""

import pytest

from ._array_module import mod as xp, mod_name
from .stubs import (
    array_attributes,
    array_methods,
    category_to_funcs,
    extension_to_funcs,
    EXTENSIONS,
)

has_name_params = []
for ext, stubs in extension_to_funcs.items():
    for stub in stubs:
        has_name_params.append(pytest.param(ext, stub.__name__))
for cat, stubs in category_to_funcs.items():
    for stub in stubs:
        has_name_params.append(pytest.param(cat, stub.__name__))
for meth in array_methods:
    has_name_params.append(pytest.param("array_method", meth.__name__))
for attr in array_attributes:
    has_name_params.append(pytest.param("array_attribute", attr))


@pytest.mark.parametrize("category, name", has_name_params)
def test_has_names(category, name):
    if category in EXTENSIONS:
        ext_mod = getattr(xp, category)
        assert hasattr(
            ext_mod, name
        ), f"{mod_name} is missing the {category} extension function {name}()"
    elif category.startswith("array_"):
        # TODO: This would fail if ones() is missing.
        arr = xp.ones((1, 1))
        if category == "array_attribute":
            assert hasattr(
                arr, name
            ), f"The {mod_name} array object is missing the attribute {name}"
        else:
            assert hasattr(
                arr, name
            ), f"The {mod_name} array object is missing the method {name}()"
    else:
        assert hasattr(
            xp, name
        ), f"{mod_name} is missing the {category} function {name}()"
