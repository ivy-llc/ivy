Ivy as a Framework
==================

On the `Building Blocks <building_blocks.rst>`_ page, we explored the role of the Backend functional APIs, the Ivy functional API, the Backend handler, and the Tracer.
These are parts labeled as (a) in the image below.

On the `Ivy as a Transpiler <ivy_as_a_transpiler.rst>`_ page, we explained the role of the backend-specific frontends in Ivy, and how these enable automatic code conversions between different ML frameworks.
This part is labeled as (b) in the image below.

So far, by considering parts (a) and (b), we have mainly treated Ivy as a fully functional framework with code conversion abilities.
Ivy builds on these primitives to create a fully-fledged ML framework with stateful classes, optimizers, and convenience tools to get ML experiments running in very few lines of code.

Specifically, here we consider the :class:`ivy.Container` class, the :class:`ivy.Array` class and the stateful API.
These parts are labeled as (c) in the image below.

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/design/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

You may choose from the following upcoming discussions or click next.

| (a) `Ivy Container <ivy_as_a_framework/ivy_container.rst>`_
| Hierarchical container solving almost everything behind the scenes in Ivy
|
| (b) `Ivy Stateful API <ivy_as_a_framework/ivy_stateful_api.rst>`_
| Trainable Layers, Modules, Optimizers, and more built on the functional API and the Ivy Container
|
| (c) `Ivy Array <ivy_as_a_framework/ivy_array.rst>`_
| Bringing methods as array attributes to Ivy, cleaning up and simplifying code

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Ivy as a Framework

   ivy_as_a_framework/ivy_container.rst
   ivy_as_a_framework/ivy_stateful_api.rst
   ivy_as_a_framework/ivy_array.rst

**Round Up**

Hopefully, this has given you a good idea of how Ivy can be used as a fully-fledged ML framework.

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!
