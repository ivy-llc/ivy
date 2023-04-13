{% extends "top_level_toc_recursive.rst" %}

{% set ivy_module_map = {
    "ivy.stateful": "Framework classes",
    "ivy.nested_array": "Nested array",
    "ivy.utils": "Utils",
    "ivy_tests.test_ivy.helpers": "Testing",
} %}

{% block name %}{{ivy_module_map[fullname] | escape | underline}}{% endblock %}
