.. include:: ../README.rst

.. toctree::
    :hidden:
    :maxdepth: -1
    :caption: Overview

    overview/background.rst
    overview/design.rst
    overview/related_work.rst
    overview/extensions.rst
    overview/contributing.rst
    overview/deep_dive.rst
    overview/faq.rst
    overview/glossary.rst


.. toctree::
    :hidden:
    :maxdepth: -1
    :caption: Compiling and Transpiling

    compiler/compiler.rst
    compiler/transpiler.rst


.. autosummary::
  :toctree: docs/functional
  :template: top_functional_toc.rst
  :caption: API Reference
  :recursive:

  ivy.functional.ivy

.. autosummary::
  :toctree: docs/data_classes
  :template: top_data_module.rst
  :recursive:
  :hide-table:

  ivy.data_classes.array
  ivy.data_classes.container
.. autosummary::
  :toctree: docs/utilities
  :template: top_level_toc_recursive.rst
  :recursive:

  ivy.stateful
  ivy.nested_array
  ivy.utils
  ivy_tests.test_ivy.helpers
