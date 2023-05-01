{% extends "top_level_module.rst" %}

{% block toctree -%}
{% if functions %}
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
{% endif %}
{% if modules %}
.. autosummary::
   :toctree: {{name}}
   :template: top_functional_module.rst
   :recursive:
{% for module in modules %}
   {{ module }}
{%- endfor %}
{% endif %}
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
{% if not functions and not classes and not attributes and not modules %}
There are no functions in this module yet. 🚧
{% endif %}

{% endblock %}
