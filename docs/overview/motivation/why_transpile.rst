Why Transpile?
==============

The crucial advancement offered by Ivy is the ability to convert machine learning models, functions, and libraries
between any of the different frameworks in this fragmented landscape. This tool is call Ivy's transpiler.
So why would we want to transpile a ML model between frameworks?

**Interoperability**

The ability to convert models helps in integrating diverse components and libraries, enabling
developers to build more complex and feature-rich applications by combining the best tools available across different frameworks.

**Flexibility**

Developers can select their preferred framework to develop in without worrying about the downstream consequences for
deployment, or what frameworks colleagues are using.

**Collaboration**

The interoperability the transpiler provides facilitates easier collaboration and knowledge sharing within organizations,
as well as the machine learning community in general. Engineers and researchers can share models and tools without being
constrained by the framework in which the work was conducted, promoting reproducibility and accelerating
progress.

**Efficiency and Optimization**

By enabling seamless conversion, developers can leverage the strengths of different frameworks within a single project.
For example, they can prototype a model in PyTorch, known for its ease of use, and then convert it to TensorFlow for
production deployment, benefiting from TensorFlow's optimization for serving models at scale. Or similarly convert the
model to JAX for its high-performance accelerator-oriented computing.

**Legacy Integration**

Legacy codebases in frameworks or framework versions that are no longer used by an organization can easily be converted
to the preferred state-of-the-art framework, saving months of painstaking migration work.

**Roundup**

Hopefully, this has given an idea of how Ivy's transpiler can be useful for the ML community 🙂

Feel free to reach out on `discord <https://discord.gg/H3pUVDeM>`_ if you have any questions!
