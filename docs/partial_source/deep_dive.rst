Deep Dive
=========

.. _`issues`: https://github.com/unifyai/ivy/issues
.. _`pull-requests`: https://github.com/unifyai/ivy/pulls

For general users of the framework, who are mainly concerned with learning how to *use* Ivy, then the :ref:`Design` section is the best place to start 🙂

This *deep dive* section is more targeted at people who would like to dive deeper into how Ivy actually works under the hood 🔧

Going through the sections outlined below will get you right into the weeds of the framework 🌱, and hopefully give you a better understanding of what is actually going on behind the scenes 🎬

It's best to go through the sub-sections from start to finish, but you can also dive in at any stage!
We're excited for you to get involved! 🦾

| (a) :ref:`Navigating the Code` 🧭
| A quick tour through the codebase
|
| (b) :ref:`Function Types` 🧮
| Primary, compositional, mixed and nestable functions
|
| (c) :ref:`Superset Behaviour` ⊃
| Ivy goes for the superset when unifying the backend functions
|
| (d) :ref:`Backend Setting` ⚙
| How the backend is set, and what this means for each function type️
|
| (e) :ref:`Arrays` 🔢
| Different types of arrays, and how they're handled
|
| (f) :ref:`Containers` 🗂
| What the :class:`ivy.Container` does
|
| (g) :ref:`Data Types` 💾
| How functions infer the correct data type
|
| (h) :ref:`Devices` 📱
| How functions infer the correct device
|
| (i) :ref:`Inplace Updates` 🎯
| How the :code:`out` argument is used to specify the output target
|
| (j) :ref:`Function Wrapping` 🎁
| How functions are dynamically wrapped at runtime
|
| (k) :ref:`Formatting` 📋
| How the code is automatically formatted
|
| (l) :ref:`Function Arguments` 📑
| How to add the correct function arguments
|
| (m) :ref:`Docstrings` 📄
| How to properly write docstrings
|
| (n) :ref:`Docstring Examples` 💯
| How to add useful examples to the docstrings
|
| (o) :ref:`Array API Tests` 🤝
| How we're borrowing the test suite from the Array API Standard
|
| (p) :ref:`Ivy Tests` 🧪
| How to add new tests for each Ivy function
|
| (q) :ref:`Ivy Frontends` ➡
| How to implement frontend functions
|
| (r) :ref:`Ivy Frontend Tests` 🧪
| How to add new tests for each frontend function
|
| (s) :ref:`Exception Handling` ⚠
| How to handle exceptions and assertions in a function
|
| (t) :ref:`Continuous Integration` 🔁
| Ivy Tests running on the Repository
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
   deep_dive/function_arguments.rst
   deep_dive/docstrings.rst
   deep_dive/docstring_examples.rst
   deep_dive/array_api_tests.rst
   deep_dive/ivy_tests.rst
   deep_dive/ivy_frontends.rst
   deep_dive/ivy_frontends_tests.rst
   deep_dive/exception_handling.rst
   deep_dive/continuous_integration.rst
