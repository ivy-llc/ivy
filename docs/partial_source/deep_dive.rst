Deep Dive
=========

.. _`issues`: https://github.com/unifyai/ivy/issues
.. _`discussions`: https://github.com/unifyai/ivy/discussions
.. _`pull-requests`: https://github.com/unifyai/ivy/pulls

For general users of the framework, who are mainly concerned with learning how to *use* Ivy,
then the :ref:`Design` section is the best place to start 🙂

This *deep dive* section is more targeted at people who would like to dive deeper into
how Ivy actually works under the hood 🔧

Going through the sections outlined below will get you right into the weeds of the framework 🌱,
and hopefully give you a better understanding of what is actually going on behind the scenes 🎬

It's best to go through the sub-sections from start to finish, but you can also dive in at any stage!
We're excited for you to get involved!  🦾

| (a) :ref:`Navigating the Code`
| A quick tour through the codebase 🔍
|
| (b) :ref:`Function Types`
| Primary, compositional, mixed and nestable functions  🧮
|
| (c) :ref:`Backend Setting`
| How the backend is set, and what this means for each function type ⚙️
|
| (d) :ref:`Function Wrapping`
| How functions are dynamically wrapped at runtime  🎁
|
| (e) :ref:`Arrays`
| Different types of arrays, and how they're handled 🔢
|
| (f) :ref:`Containers`
| What the :code:`ivy.Container` does  🗂️
|
| (g) :ref:`Data Types`
| How functions infer the correct data type  💾
|
| (h) :ref:`Devices`
| How functions infer the correct device  💽
|
| (i) :ref:`Inplace Updates`
| How the :code:`out` argument is used to specify the output target  🎯
|
| (j) :ref:`Formatting`
| How the code is automatically formatted 📋
|
| (k) :ref:`Function Arguments`
| How to add the correct function arguments 📑
|
| (l) :ref:`Docstrings`
| How to properly write docstrings 📄
|
| (m) :ref:`Docstring Examples`
| How to add useful examples to the docstrings 💯
|
| (n) :ref:`Array API Tests`
| How we're borrowing the test suite from the Array API Standard 🤝
|
| (o) :ref:`Ivy Tests`
| How to add new tests for each Ivy function ❓
|
| (p) :ref:`Ivy Frontends`
| How to implement frontend functions ➡️
|
| (q) :ref:`Ivy Frontend Tests`
| How to add new tests for each frontend function ➡️❓

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Deep Dive

   deep_dive/0_navigating_the_code.rst
   deep_dive/1_function_types.rst
   deep_dive/2_backend_setting.rst
   deep_dive/3_function_wrapping.rst
   deep_dive/4_arrays.rst
   deep_dive/5_containers.rst
   deep_dive/6_data_types.rst
   deep_dive/7_devices.rst
   deep_dive/8_inplace_updates.rst
   deep_dive/9_formatting.rst
   deep_dive/10_function_arguments.rst
   deep_dive/11_docstrings.rst
   deep_dive/12_docstring_examples.rst
   deep_dive/13_array_api_tests.rst
   deep_dive/14_ivy_tests.rst
   deep_dive/15_ivy_frontends.rst
   deep_dive/16_ivy_frontends_tests.rst
