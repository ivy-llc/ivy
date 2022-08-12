Design
======

This section is aimed at general users, who would like to learn how to use Ivy,
and are less concerned about how it all works under the hood 🔧

The :ref:`Deep Dive` section is more targeted at potential contributors,
and at users who would like to dive deeper into the weeds of the framework🌱,
and gain a better understanding of what is actually going on behind the scenes 🎬

If that sounds like you, feel free to check out the :ref:`Deep Dive` section
after you've gone through the higher level overview which is covered in this *design* section!

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

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/dependency_graph_with_compiler.png?raw=true
   :align: center
   :width: 100%

| (a) :ref:`Building Blocks`
| back-end functional APIs ✅
| Ivy functional API ✅
| Framework Handler ✅
| Ivy Compiler 🚧
|
| (b) :ref:`Ivy as a Transpiler`
| front-end functional APIs 🚧
|
| (c) :ref:`Ivy as a Framework`
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
