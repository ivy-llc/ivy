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

.. autosummary::
  :toctree: docs/functional
  :template: top_level_toc.rst
  :recursive:
  :include:

  ivy.functional.ivy


.. autosummary::
  :toctree: docs/data_classes
  :caption: Data classes
  :template: top_data_module.rst
  :recursive:
  :hide-table:

  ivy.data_classes.array
  ivy.data_classes.container

.. autosummary::
  :toctree: docs/framework
  :template: top_level_toc.rst
  :recursive:
  :include:

  ivy.stateful

.. autosummary::
  :toctree: docs/nested-array
  :template: top_level_toc.rst
  :recursive:
  :include:

  ivy.nested_array

.. autosummary::
  :toctree: docs/utilities
  :template: top_level_toc.rst
  :recursive:
  :include:

  ivy.utils

.. autosummary::
  :toctree: docs/testing
  :template: top_level_toc.rst
  :recursive:
  :include:

  ivy_tests.test_ivy.helpers
