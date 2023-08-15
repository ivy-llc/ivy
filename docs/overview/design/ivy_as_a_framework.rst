Ivy as a Framework
==================

On the :ref:`Building Blocks` page, we explored the role of the backend functional APIs, the Ivy functional API, the framework handler and the graph compiler.
These are parts labeled as (a) in the image below.

On the :ref:`Ivy as a Transpiler` page, we explained the role of the backend-specific frontends in Ivy, and how these enable automatic code conversions between different ML frameworks.
This part is labeled as (b) in the image below.

So far, by considering parts (a) and (b), we have mainly treated Ivy as a fully functional framework with code conversion abilities.
Ivy builds on these primitives to create a fully-fledged ML framework with stateful classes, optimizers and convenience tools to get ML experiments running in very few lines of code.

Specifically, here we consider the :class:`ivy.Container` class, the :class:`ivy.Array` class and the stateful API.
These parts are labeled as (c) in the image below.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

You may choose from the following upcoming discussions or click next.

| (a) :ref:`Ivy Container`
| Hierarchical container solving almost everything behind the scenes in Ivy
|
| (b) :ref:`Ivy Stateful API`
| Trainable Layers, Modules, Optimizers and more built on the functional API and the Ivy Container
|
| (c) :ref:`Ivy Array`
| Bringing methods as array attributes to Ivy, cleaning up and simplifying code

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Ivy as a Framework

   ivy_as_a_framework/ivy_container.rst
   ivy_as_a_framework/ivy_stateful_api.rst
   ivy_as_a_framework/ivy_array.rst

**Round Up**

Hopefully this has given you a good idea of how Ivy can be used as a fully-fledged ML framework.

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!
