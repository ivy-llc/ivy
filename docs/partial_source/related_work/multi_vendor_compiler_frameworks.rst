Multi-Vendor Compiler Frameworks
================================

.. _`Tensor Virtual Machine (TVM)`: https://tvm.apache.org/
.. _`actively exploring`: https://discuss.tvm.apache.org/t/google-lasted-work-mlir-primer/1721
.. _`MLIR`: https://mlir.llvm.org/
.. _`Accelerated Linear Algebra (XLA)`: https://www.tensorflow.org/xla
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`JAX`: https://jax.readthedocs.io/
.. _`PyTorch`: https://pytorch.org/
.. _`Julia`: https://julialang.org/

The compiler frameworks explained below enable Machine Learning code to be executed on a variety of hardware targets, with abstractions selected carefully in order to simplify this process and reduce the implementational overhead for supporting many different end targets. In general, these multi-target compiler frameworks can also make use of compiler infrastructure such as that explained in the previous section, in order to follow best practices, streamline the design, and maximize interoperability.

Apache TVM
----------
Apache's `Tensor Virtual Machine (TVM)`_ is an open source machine learning compiler framework for CPUs, GPUs, and machine learning accelerators which aims to enable machine learning engineers to optimize and run computations efficiently on any hardware backend. It enables the compilation of deep learning models into minimum deployable modules, and it provides infrastructure to automatically generate and optimize models on more backends with better performance. Apache TVM is an incredibly useful framework, which simplifies Machine Learning deployment to various hardware vendors. TVM is `actively exploring`_ the potential integration of `MLIR`_ principles into the design.

XLA
---
`Accelerated Linear Algebra (XLA)`_ is a compiler for linear algebra that can accelerate models with potentially no source code changes. The results are improvements in speed and memory usage. Conventionally, when ML programs are run, all of the operations are executed individually on the target device. In the case of GPU execution, each operation has a precompiled GPU kernel implementation that the executor dispatches to. XLA provides an alternative mode of running models: it compiles the graph into a sequence of computation kernels generated specifically for the given model. Because these kernels are unique to the model, they can exploit model-specific information for optimization. XLA is supported by `TensorFlow`_, `JAX`_, `PyTorch`_ and the `Julia`_ language, and is able to compile to TPUs, GPUs and CPUs.
