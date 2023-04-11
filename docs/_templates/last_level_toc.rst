{{name | underline}}

.. This is a placeholder so the include directive removes what's before it
.. REMOVE_BEFORE_HERE
.. autosummary::
   :toctree: {{name}}
   :template: {% block template %}top_level_module.rst{% endblock %}
   :hide-table:
   :recursive:

   {{ fullname }}
