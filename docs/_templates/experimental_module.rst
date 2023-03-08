{{ name | replace("_", " ") | capitalize | escape | underline }}

.. autosummary::
   :toctree: {{name}}
   :template: experimental_submodule.rst
   :include:

{% for submodule in modules %}   {{ submodule }}
{% endfor %}