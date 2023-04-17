{% extends "top_level_module.rst" %}

{% block toctree -%}
.. autosummary::
   :toctree: {{name}}
   :template: functional_module.rst
   :hide-table:
   :recursive:
{% for function in functions %}
   {% if not function.startswith('_') %}
   {{ fullname }}.{{ function }}
   {% endif %}
{%- endfor %}

{% endblock %}

{% block options %}
   :special-members: __init__
   :show-inheritance:
{% endblock %}

{% block custom_content %}
{% for function in functions %}
.. autofunction:: ivy.{{ function }}
{% endfor %}
{% for class in classes %}
.. autoclass:: ivy.{{ class }}
{% endfor %}
{% for attribute in attributes %}
.. autoivydata:: {{ fullname }}.{{ attribute }}
{% endfor %}

{% endblock %}
