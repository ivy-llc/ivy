Deep Dive
=========

.. _`issues`: https://github.com/unifyai/ivy/issues
.. _`pull-requests`: https://github.com/unifyai/ivy/pulls

For general users of the framework, who are mainly concerned with learning how to *use* Ivy, then the `Design <design.rst>`_ section is the best place to start 🙂

This *deep dive* section is more targeted at people who would like to dive deeper into how Ivy actually works under the hood 🔧

Going through the sections outlined below will get you right into the weeds of the framework 🌱, and hopefully give you a better understanding of what is actually going on behind the scenes 🎬

It's best to go through the sub-sections from start to finish, but you can also dive in at any stage!
We're excited for you to get involved! 🦾

| (a) `Navigating the Code <deep_dive/navigating_the_code.rst>`_ 🧭
| A quick tour through the codebase
|
| (b) `Function Types <deep_dive/function_types.rst>`_ 🧮
| Primary, compositional, mixed, and nestable functions
|
| (c) `Superset Behaviour <deep_dive/superset_behaviour.rst>`_ ⊃
| Ivy goes for the superset when unifying the backend functions
|
| (d) `Backend Setting <deep_dive/backend_setting.rst>`_ ⚙
| How the backend is set, and what this means for each function type️
|
| (e) `Arrays <deep_dive/arrays.rst>`_ 🔢
| Different types of arrays, and how they're handled
|
| (f) `Containers <deep_dive/containers.rst>`_ 🗂
| What the :class:`ivy.Container` does
|
| (g) `Data Types <deep_dive/data_types.rst>`_ 💾
| How functions infer the correct data type
|
| (h) `Devices <deep_dive/devices.rst>`_ 📱
| How functions infer the correct device
|
| (i) `Inplace Updates <deep_dive/inplace_updates.rst>`_ 🎯
| How the :code:`out` argument is used to specify the output target
|
| (j) `Function Wrapping <deep_dive/function_wrapping.rst>`_ 🎁
| How functions are dynamically wrapped at runtime
|
| (k) `Formatting <deep_dive/formatting.rst>`_ 📋
| How the code is automatically formatted
|
| (l) `Ivy Lint <deep_dive/ivy_lint.rst>`_ 🧹
| Ivy's Custom Code Formatters
|
| (m) `Function Arguments <deep_dive/function_arguments.rst>`_ 📑
| How to add the correct function arguments
|
| (n) `Docstrings <deep_dive/docstrings.rst>`_ 📄
| How to properly write docstrings
|
| (o) `Docstring Examples <deep_dive/docstring_examples.rst>`_ 💯
| How to add useful examples to the docstrings
|
| (p) `Array API Tests <deep_dive/array_api_tests.rst>`_ 🤝
| How we're borrowing the test suite from the Array API Standard
|
| (q) `Ivy Tests <deep_dive/ivy_tests.rst>`_ 🧪
| How to add new tests for each Ivy function
|
| (r) `Ivy Frontends <deep_dive/ivy_frontends.rst>`_ ➡
| How to implement frontend functions
|
| (s) `Ivy Frontend Tests <deep_dive/ivy_frontends_tests.rst>`_ 🧪
| How to add new tests for each frontend function
|
| (t) `Exception Handling <deep_dive/exception_handling.rst>`_ ⚠
| How to handle exceptions and assertions in a function
|
| (u) `Continuous Integration <deep_dive/continuous_integration.rst>`_ 🔁
| Ivy Tests running on the Repository
|
| (v) `Gradients <deep_dive/gradients.rst>`_ 🔁
| Everything about our Gradients API
|
| (w) `Operating Modes <deep_dive/operating_modes.rst>`_ 🧮
| Everything about modes Ivy can operate in, along with their purposes
|
| (x) `Building the Docs Pipeline <deep_dive/building_the_docs_pipeline.rst>`_ 📚
| How are we building our docs


.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Deep Dive

   deep_dive/navigating_the_code.rst
   deep_dive/function_types.rst
   deep_dive/superset_behaviour.rst
   deep_dive/backend_setting.rst
   deep_dive/arrays.rst
   deep_dive/containers.rst
   deep_dive/data_types.rst
   deep_dive/devices.rst
   deep_dive/inplace_updates.rst
   deep_dive/function_wrapping.rst
   deep_dive/formatting.rst
   deep_dive/ivy_lint.rst
   deep_dive/function_arguments.rst
   deep_dive/docstrings.rst
   deep_dive/docstring_examples.rst
   deep_dive/array_api_tests.rst
   deep_dive/ivy_tests.rst
   deep_dive/ivy_frontends.rst
   deep_dive/ivy_frontends_tests.rst
   deep_dive/exception_handling.rst
   deep_dive/continuous_integration.rst
   deep_dive/gradients.rst
   deep_dive/operating_modes.rst
   deep_dive/building_the_docs_pipeline.rst
   deep_dive/fix_failing_tests.rst
