Related Work
============

.. _`RWorks API Standards`: related_work/api_standards.rst
.. _`RWorks Wrapper Frameworks`: related_work/wrapper_frameworks.rst
.. _`RWorks Frameworks`: related_work/frameworks.rst
.. _`RWorks Graph Tracers`: related_work/graph_tracers.rst
.. _`RWorks Exchange Formats`: related_work/exchange_formats.rst
.. _`RWorks Compiler Infrastructure`: related_work/compiler_infrastructure.rst
.. _`RWorks Multi-Vendor Compiler Frameworks`: related_work/multi_vendor_compiler_frameworks.rst
.. _`RWorks Vendor-Specific APIs`: related_work/vendor_specific_apis.rst
.. _`RWorks Vendor-Specific Compilers`: related_work/vendor_specific_compilers.rst
.. _`RWorks ML-Unifying Companies`: related_work/ml_unifying_companies.rst
.. _`RWorks What does Ivy Add?`: related_work/what_does_ivy_add.rst

In this section, we explain how Ivy compares to many other very important and related pieces of work, which also address fragmentation but at other areas within the ML stack.

Firstly, we need to look at the overall ML stack, and understand how the high level frameworks relate to the low level components.

In order to conceptualize this rather complex hierarchy, we have broken the ML stack into 9 groups, which are: `RWorks API Standards`_, `RWorks Wrapper Frameworks`_, `RWorks Frameworks`_, `RWorks Graph Tracers`_, `RWorks Exchange Formats`_, `RWorks Compiler Infrastructure`_, `RWorks Multi-Vendor Compiler Frameworks`_, `RWorks Vendor-Specific APIs`_ and `RWorks Vendor-Specific Compilers`_, going from high level to low level respectively.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/related_work/ml_stack.png?raw=true
   :width: 100%

Each of these groups within the ML stack has it's own sub-section, linked below, within which we discuss various related projects which operate at that particular level within the stack.

We then compare Ivy to some other ML-unifying companies which are working on very important problems and are helping to unify the lower levels of the ML stack.
We see these efforts as being very complimentary to Ivy's vision for high level unification.

Finally, we discuss how Ivy compares to each of these important works at all levels within the ML stack.


| (a) `RWorks API Standards`_ 🤝🏽
| Standardized APIs which similar libraries should adhere to
|
| (b) `RWorks Wrapper Frameworks`_ 🎁
| Frameworks which wrap other ML frameworks
|
| (c) `RWorks Frameworks`_ 🔢
| Standalone ML Frameworks
|
| (d) `RWorks Graph Tracers`_ 🕸️
| Extracting acyclic directed computation graphs from code
|
| (e) `RWorks Exchange Formats`_ 💱
| File formats to exchange neural networks between frameworks
|
| (f) `RWorks Compiler Infrastructure`_ 🔟️🏗️
| Infrastructure and standards to simplify the lives of compiler designers
|
| (g) `RWorks Multi-Vendor Compiler Frameworks`_ 🖥️💻🔟
| Executing ML code on a variety of hardware targets
|
| (h) `RWorks Vendor-Specific APIs`_ 💻🔢
| Interfacing with specific hardware in an intuitive manner
|
| (i) `RWorks Vendor-Specific Compilers`_ 💻🔟
| Compiling code to specific hardware
|
| (j) `RWorks ML-Unifying Companies`_ 📈
| Companies working towards unification in ML
|
| (k) `RWorks What does Ivy Add?`_ 🟢
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
