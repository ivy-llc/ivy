.. include:: ../README.rst
    :end-before: <br clear="all" />

.. raw:: html
    
    <br clear="all" />
    <br/>
    <br/>

.. include:: ../README.rst
    :start-after: <br clear="all" />
    :end-before: Check out the docs_ for more info

Check out our Google Colabs_ for some interactive demos!

.. include:: ../README.rst
    :start-after: and check out our Google Colabs_ for some interactive demos!

.. toctree::
    :hidden:
    :maxdepth: -1
    :caption: Overview

    partial_source/background.rst
    partial_source/design.rst
    partial_source/related_work.rst
    partial_source/extensions.rst
    partial_source/contributing.rst
    partial_source/deep_dive.rst
    partial_source/faq.rst
    partial_source/glossary.rst


.. .. autosummary
..   :toctree: docs/functional
..   :caption: Functions
..   :recursive:


.. .. autosummary
..   :toctree: docs/data_classes
..   :caption: Data classes
..   :template: top_level_module.rst
..   :recursive:
..   :hide-table:

..   ivy.array.array

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
