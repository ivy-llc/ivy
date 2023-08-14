Related Work
============

In this section, we explain how Ivy compares to many other very important and related pieces of work, which also address fragmentation but at other areas within the ML stack.

Firstly, we need to look at the overall ML stack, and understand how the high level frameworks relate to the low level components.

In order to conceptualize this rather complex hierarchy, we have broken the ML stack into 9 groups, which are: :ref:`RWorks API Standards`, :ref:`RWorks Wrapper Frameworks`, :ref:`RWorks Frameworks`, :ref:`RWorks Graph Tracers`, :ref:`RWorks Exchange Formats`, :ref:`RWorks Compiler Infrastructure`, :ref:`RWorks Multi-Vendor Compiler Frameworks`, :ref:`RWorks Vendor-Specific APIs` and :ref:`RWorks Vendor-Specific Compilers`, going from high level to low level respectively.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/related_work/ml_stack.png?raw=true
   :width: 100%

Each of these groups within the ML stack has it's own sub-section, linked below, within which we discuss various related projects which operate at that particular level within the stack.

We then compare Ivy to some other ML-unifying companies which are working on very important problems and are helping to unify the lower levels of the ML stack.
We see these efforts as being very complimentary to Ivy's vision for high level unification.

Finally, we discuss how Ivy compares to each of these important works at all levels within the ML stack.


| (a) :ref:`RWorks API Standards` 🤝🏽
| Standardized APIs which similar libraries should adhere to
|
| (b) :ref:`RWorks Wrapper Frameworks` 🎁
| Frameworks which wrap other ML frameworks
|
| (c) :ref:`RWorks Frameworks` 🔢
| Standalone ML Frameworks
|
| (d) :ref:`RWorks Graph Tracers` 🕸️
| Extracting acyclic directed computation graphs from code
|
| (e) :ref:`RWorks Exchange Formats` 💱
| File formats to exchange neural networks between frameworks
|
| (f) :ref:`RWorks Compiler Infrastructure` 🔟️🏗️
| Infrastructure and standards to simplify the lives of compiler designers
|
| (g) :ref:`RWorks Multi-Vendor Compiler Frameworks` 🖥️💻🔟
| Executing ML code on a variety of hardware targets
|
| (h) :ref:`RWorks Vendor-Specific APIs` 💻🔢
| Interfacing with specific hardware in an intuitive manner
|
| (i) :ref:`RWorks Vendor-Specific Compilers` 💻🔟
| Compiling code to specific hardware
|
| (j) :ref:`RWorks ML-Unifying Companies` 📈
| Companies working towards unification in ML
|
| (k) :ref:`RWorks What does Ivy Add?` 🟢
| How does Ivy fit into all of this?

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Related Work

   related_work/api_standards.rst
   related_work/wrapper_frameworks.rst
   related_work/frameworks.rst
   related_work/graph_tracers.rst
   related_work/exchange_formats.rst
   related_work/compiler_infrastructure.rst
   related_work/multi_vendor_compiler_frameworks.rst
   related_work/vendor_specific_apis.rst
   related_work/vendor_specific_compilers.rst
   related_work/ml_unifying_companies.rst
   related_work/what_does_ivy_add.rst
