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
.. _`GNU Compiler Collection (GCC)`: https://gcc.gnu.org/git/gcc.git
.. _`GNU Project`: https://www.gnu.org/
.. _`Free Software Foundation (FSF)`: https://www.fsf.org/
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`related work channel`: https://discord.com/channels/799879767196958751/1034436036371157083
.. _`related work forum`: https://discord.com/channels/799879767196958751/1034436085587120149

The compiler frameworks explained below enable Machine Learning code to be executed on a variety of hardware targets, with abstractions selected carefully in order to simplify this process and reduce the implementational overhead for supporting many different end targets.
In general, these multi-target compiler frameworks can also make use of compiler infrastructure such as that explained in the previous section, in order to follow best practices, streamline the design, and maximize interoperability.

Apache TVM
----------
Apache's `Tensor Virtual Machine (TVM)`_ is an open source machine learning compiler framework for CPUs, GPUs, and machine learning accelerators which aims to enable machine learning engineers to optimize and run computations efficiently on any hardware backend.
It enables the compilation of deep learning models into minimum deployable modules, and it provides infrastructure to automatically generate and optimize models on more backends with better performance.
Apache TVM is an incredibly useful framework, which simplifies Machine Learning deployment to various hardware vendors.
TVM is `actively exploring`_ the potential integration of `MLIR`_ principles into the design.

XLA
---
`Accelerated Linear Algebra (XLA)`_ is a compiler for linear algebra that can accelerate models with potentially no source code changes.
The results are improvements in speed and memory usage.
Conventionally, when ML programs are run, all of the operations are executed individually on the target device.
In the case of GPU execution, each operation has a precompiled GPU kernel implementation that the executor dispatches to.
XLA provides an alternative mode of running models: it compiles the graph into a sequence of computation kernels generated specifically for the given model.
Because these kernels are unique to the model, they can exploit model-specific information for optimization.
XLA is supported by `TensorFlow`_, `JAX`_, `PyTorch`_ and the `Julia`_ language, and is able to compile to TPUs, GPUs and CPUs.

GCC
---

The `GNU Compiler Collection (GCC)`_ is an optimizing compiler produced by the `GNU Project`_ supporting various programming languages, hardware architectures and operating systems.
The `Free Software Foundation (FSF)`_ distributes GCC as free software under the GNU General Public License (GNU GPL).
GCC is a key component of the GNU toolchain and the standard compiler for most projects related to GNU and the Linux kernel.
With roughly 15 million lines of code in 2019, GCC is one of the biggest free programs in existence, and it has played an important role in the growth of free software, as both a tool and an example.

**Round Up**

If you have any questions, please feel free to reach out on `discord`_ in the `related work channel`_ or in the `related work forum`_!