.. _`RWorks Vendor-Specific APIs`:

Vendor-Specific APIs
====================

.. _`CUDA`: https://developer.nvidia.com/cuda-toolkit
.. _`TensorRT`: https://developer.nvidia.com/tensorrt
.. _`NVIDIA`: https://www.nvidia.com/
.. _`PyTorch`: https://pytorch.org/
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Compute Unified Device Architecture (CUDA)`: https://developer.nvidia.com/cuda-toolkit
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`related work channel`: https://discord.com/channels/799879767196958751/1034436036371157083
.. _`related work forum`: https://discord.com/channels/799879767196958751/1034436085587120149

Vendor-specific APIs provide an interface to define customized operations for hardware from specific vendors.
The libraries are written exclusively for hardware from this vendor, and so the code is clearly not generalized nor is it intended to be.
These APIs are often used by higher level multi-vendor compilers and frameworks, and most machine learning practitioners will not interface with these low level vendor-specific APIs directly.

TensorRT
--------
Built on top of `CUDA`_, `TensorRT`_ is a C++ library for high performance inference on `NVIDIA`_ GPUs and deep learning accelerators.
It is integrated with `PyTorch`_ and TensorFlow.
When conducting deep learning training in a proprietary or custom framework, then the TensorRT C++ API can be used to import and accelerate models.
Several optimizations contribute to the high performance: reduced mixed precision maximizes throughput, layer and tensor fusion optimizes device memory, kernel autotuning selects the best data layers and algorithms, time fusion optimizes recurrent neural networks, multi-stream execution manages input streams, and dynamic tensor memory optimizes memory consumption.

CUDA
----
`Compute Unified Device Architecture (CUDA)`_ is a parallel computing platform and application programming interface (API) that allows software to use certain types of graphics processing units (GPUs) for general purpose processing, an approach called general-purpose computing on GPUs (GPGPU).
It is a software layer that gives direct access to the GPU's virtual instruction set and parallel computational elements, for the execution of compute kernels.
It is designed to work with programming languages such as C, C++, and Fortran.

**Round Up**

If you have any questions, please feel free to reach out on `discord`_ in the `related work channel`_ or in the `related work forum`_!