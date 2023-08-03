.. title:: Home

.. include:: ../README.rst

.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: Overview

  overview/get_started.rst
  Examples <https://unify.ai/demos/>
  overview/glossary.rst
  overview/faq.rst


.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: Users

  overview/background.rst
  overview/design.rst
  overview/related_work.rst
  overview/extensions.rst


.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: Contributors

  overview/deep_dive.rst
  overview/contributing.rst


.. toctree::
  :hidden:
  :maxdepth: -1
  :caption: Compiling and Transpiling

  compiler/setting_up.rst
  compiler/compiler.rst
  compiler/transpiler.rst


.. autosummary::
  :toctree: docs/functional
  :template: top_functional_toc.rst
  :caption: API Reference
  :recursive:
  :hide-table:

  ivy.functional.ivy


.. autosummary::
  :toctree: docs/data_classes
  :template: top_data_toc.rst
  :recursive:
  :hide-table:

  ivy.data_classes


.. autosummary::
  :toctree: docs
  :template: top_ivy_toc.rst
  :recursive:
  :hide-table:

  ivy.stateful
  ivy.utils
  ivy_tests.test_ivy.helpers
