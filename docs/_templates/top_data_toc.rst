{% extends "top_level_toc.rst" %}
{% block name %}{{"Data classes" | escape | underline}}{% endblock %}

{% block template %}top_data_module.rst{% endblock %}

{% block options %}{{ super() }}    :hide-table:
{% endblock %}

{#
    As this toc generates files a little differently, we added this to fix linking
    issues
#}
{% block custom_content %}
.. autosummary::

{% for submodule in modules %}
    {{ submodule }}.{{ submodule.split('.')[-1] }}
{%- endfor %}

{% endblock %}
