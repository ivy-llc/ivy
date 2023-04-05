{% extends "top_level_module.rst" %}

{% block toctree -%}
.. autosummary::
   :toctree: {{name}}
   :template: functional_module.rst
   :hide-table:
   :recursive:
{% for function in functions %}
   {{ fullname }}.{{ function }}
{%- endfor %}

{% endblock %}
