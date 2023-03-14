{{name | underline}}

.. This is a placeholder so the include directive removes what's before it
.. REMOVE_BEFORE_HERE
.. autosummary::
   :toctree: {{name}}
   :template: top_level_module.rst
   :caption: {{fullname}}
   :substitute-caption:
   :hide-table:
   :fix-directory:
{% for submodule in modules %}
   {{ submodule }}
{%- endfor %}