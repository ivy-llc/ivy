{{ name | replace("_", " ") | capitalize | escape | underline }}

{% block toctree %}{% endblock %}

.. automodule:: {% block module_name %}{{fullname}}{% endblock %}{% block options %}
    :members:
    :special-members: __init__
    :undoc-members:
    :show-inheritance:
{% endblock %}

{% block custom_content %}{% endblock %}

.. discussion-links:: {% block discussion_module_name %}{{fullname}}{% endblock %}
