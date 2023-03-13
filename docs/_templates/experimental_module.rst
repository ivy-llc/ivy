{{ name | replace("_", " ") | capitalize | escape | underline }}

{% for submodule in modules %}
.. automodule:: {{submodule}}
    :members:
    :special-members: __init__
    :undoc-members:
{% endfor %}