{{ name | replace("_", " ") | capitalize | escape | underline }}

{% block toctree %}{% endblock %}

.. automodule:: {% block module_name %}{{fullname}}{% endblock %}
    :members:
    :special-members: __init__
    :undoc-members:
    :show-inheritance:

.. discussion-links:: {% block discussion_module_name %}{{fullname}}{% endblock %}
