{% extends "top_level_module.rst" %}

{%- block options -%}
    {{super()}}    :private-members:
{%- endblock -%}

.. Experimental modules are added here
{% block custom_content %}
{% for submodule in modules %}
.. automodule:: {{submodule}}
    :members:
    :special-members: __init__
    :undoc-members:
    :private-members:

{% endfor %}
{% endblock %}
