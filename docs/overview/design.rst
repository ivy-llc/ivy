Design
======

.. _`Deep Dive`: deep_dive.rst

This section is aimed at general users, who would like to learn how to use Ivy, and are less concerned about how it all works under the hood 🔧

The `Deep Dive`_ section is more targeted at potential contributors, and at users who would like to dive deeper into the weeds of the framework🌱, and gain a better understanding of what is actually going on behind the scenes 🎬

If that sounds like you, feel free to check out the `Deep Dive`_ section after you've gone through the higher level overview which is covered in this *design* section!

| So, starting off with our higher level *design* section, Ivy can fulfill two distinct purposes:
|
| 1. enable automatic code conversions between frameworks
| 2. serve as a new ML framework with multi-framework support
|
| The Ivy codebase can then be split into three categories which are labelled (a),
| (b) and (c) below, and can be further split into 8 distinct submodules.
| The eight submodules are Ivy API, Backend Handler, Backend API, Ivy Array,
| Ivy Container, Ivy Stateful API, and finally Frontend API.

| All eight fall into one of the three categories as follows:

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

| (a) `Building Blocks <design/building_blocks.rst>`_
| back-end functional APIs ✅
| Ivy functional API ✅
| Framework Handler ✅
| Ivy Tracer 🚧
|
| (b) `Ivy as a Transpiler <design/ivy_as_a_transpiler.rst>`_
| front-end functional APIs 🚧
|
| (c) `Ivy as a Framework <design/ivy_as_a_framework.rst>`_
| Ivy stateful API ✅
| Ivy Container ✅
| Ivy Array ✅

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Design

   design/building_blocks.rst
   design/ivy_as_a_transpiler.rst
   design/ivy_as_a_framework.rst
