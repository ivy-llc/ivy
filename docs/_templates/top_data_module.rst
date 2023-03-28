{% extends "top_level_module.rst" %}

{% set base_module = fullname + "." + fullname.split('.')[-1] %}

{%- block module_name -%}
    {{base_module}}
{%- endblock -%}

{%- block discussion_module_name -%}
    {{base_module}}
{%- endblock -%}

{% block toctree -%}
.. autosummary::
   :toctree: {{name}}
   :template: data_module.rst
   :hide-table:
   :recursive:
{% for submodule in modules -%}
{% if base_module != submodule %}
   {{ submodule }}
{% endif -%}
{% endfor -%}

{% endblock %}
