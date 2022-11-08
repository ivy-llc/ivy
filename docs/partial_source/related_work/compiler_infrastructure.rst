Compiler Infrastructure
=======================

.. _`LLVM`: https://llvm.org/
.. _`Multi Level Intermediate Representation (MLIR)`: https://mlir.llvm.org/
.. _`MLIR`: https://mlir.llvm.org/
.. _`Onnx-mlir`: https://github.com/onnx/onnx-mlir
.. _`ONNX`: https://onnx.ai/
.. _`OneAPI`: https://www.oneapi.io/
.. _`Intel`: https://www.intel.com/
.. _`OneDNN`: https://github.com/oneapi-src/oneDNN
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`related work channel`: https://discord.com/channels/799879767196958751/1034436036371157083
.. _`related work forum`: https://discord.com/channels/799879767196958751/1034436085587120149

Compiler infrastructure generally provides carefully thought through frameworks and principles to simplify the lives of compiler designers, maximizing the reusability of tools and interoperability when deploying to various different hardware targets.
This infrastructure doesn’t provide “full” solutions for compiling to hardware, but instead provides the general scaffolding to make the design of such compilers as principled and interoperable as possible, with maximal code sharing and interoperability being at the heart of their design.

LLVM
----
`LLVM`_ is a set of compiler and toolchain technologies that can be used to develop a front end for any programming language and a back end for any instruction set architecture.
LLVM is designed around a language-independent intermediate representation (IR) that serves as a portable, high-level assembly language that can be optimized with a variety of transformations over multiple passes.
It is designed for compile-time, link-time, run-time, and "idle-time" optimization.
It can provide the middle layers of a complete compiler system, taking intermediate representation (IR) code from a compiler and emitting an optimized IR.
This new IR can then be converted and linked into machine-dependent assembly language code for a target platform.
It can also accept the IR from the GNU Compiler Collection (GCC) toolchain, allowing it to be used with a wide array of existing compiler front-ends written for that project.

MLIR
----
The `Multi Level Intermediate Representation (MLIR)`_ is an important piece of compiler infrastructure designed to represent multiple levels of abstraction, with abstractions and domain-specific IR constructs being easy to add, and with location being a first-class construct.
It is part of the broader `LLVM`_ project.
It aims to address software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain specific compilers, and aid in connecting existing compilers together.
Comparing to other parts of the overall ML stack, MLIR is designed to operate at a lower level than the neural network exchange formats.
For example, the `Onnx-mlir`_ compiler uses the MLIR compiler infrastructure to implement a compiler which enables `ONNX`_ defined models to be compiled into native code.

OneAPI
------
`OneAPI`_ is an open standard for a unified Application Programming Interface (API) intended to be used across different compute accelerator (coprocessor) architectures, including GPUs, AI accelerators and field-programmable gate arrays, although at present the main user is `Intel`_, with them being the authors of the standard.
The set of APIs spans several domains that benefit from acceleration, including libraries for linear algebra math, deep learning, machine learning, video processing, and others.
`OneDNN`_ is particularly relevant, focusing on neural networks functions for deep learning training and inference.
Intel CPUs and GPUs have accelerators for Deep Learning software, and OneDNN provides a unified interface to utilize these accelerators, with much of the hardware-specific complexity abstracted away.
In a similar manner to `MLIR`_, OneAPI is also designed to operate at a lower level than the Neural Network :ref:`Exchange Formats`.
The interface is lower level and more primitive than the neural network exchange formats, with focus on the core low-level operations such as convolutions, matrix multiplications, batch normalization etc.
This makes OneDNN very much complementary to these formats, where OneDNN can sit below the exchange formats in the overall stack, enabling accelerators to be fully leveraged with minimal hardware-specific considerations, with this all helpfully being abstracted by the OneDNN API.
Indeed, OneAPI and MLIR can work together in tandem, and OneDNN are working to `integrate Tensor Possessing Primitives in the MLIR compilers used underneath TensorFlow <https://www.oneapi.io/blog/tensorflow-and-onednn-in-partnership/>`_.

**Round Up**

If you have any questions, please feel free to reach out on `discord`_ in the `related work channel`_ or in the `related work forum`_!