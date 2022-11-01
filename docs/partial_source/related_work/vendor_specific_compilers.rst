Vendor-Specific Compilers
=========================

.. _`Intel C++ Compiler Classic (ICC)`: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference
.. _`Intel oneAPI DPC++/C++ Compiler (ICX)`: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html
.. _`Intel`: https://www.intel.com/
.. _`ICC`: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference
.. _`Khronos Group`: https://www.khronos.org/
.. _`LLVM`: https://llvm.org/
.. _`Nvidia CUDA Compiler (NVCC)`: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc
.. _`NVIDIA`: https://www.nvidia.com/
.. _`CUDA`: https://developer.nvidia.com/cuda-toolkit
.. _`GCC`: https://gcc.gnu.org/
.. _`Microsoft Visual C++ Compiler`: https://docs.microsoft.com/en-us/cpp/
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`related work channel`: https://discord.com/channels/799879767196958751/1034436036371157083
.. _`related work forum`: https://discord.com/channels/799879767196958751/1034436085587120149

Below the vendor-specific APIs are the vendor specific compilers.
As far as modern machine learning practitioners go, these compilers are very rarely interacted with directly.
As for our own representation of the ML stack, these compilers are the lowest level building blocks that we consider.
Of course, we could talk about assembly languages and byte code, but this is outside the scope of what is really relevant for ML practitioners when considering their software stack.

ICC
---
`Intel C++ Compiler Classic (ICC)`_ is the first of `Intel`_’s two C, C++, SYCL, and Data Parallel C++ (DPC++) compilers for Intel processor-based systems.
It is available for Windows, Linux, and macOS operating systems.
It targets general-purpose Intel x86-64 architecture CPUs.

ICX
---
`Intel oneAPI DPC++/C++ Compiler (ICX)`_ is the second of `Intel`_’s two C, C++, SYCL, and Data Parallel C++ (DPC++) compilers for Intel processor-based systems.
Again, it is available for Windows, Linux, and macOS operating systems.
Unlike `ICC`_, It generates code for both Intel’s general-purpose x86-64 CPUs and also GPUs.
Specifically, it targets Intel IA-32, Intel 64 (aka x86-64), Core, Xeon, and Xeon Scalable processors, as well as GPUs including Intel Processor Graphics Gen9 and above, Intel Xe architecture, and Intel Programmable Acceleration Card with Intel Arria 10 GX FPGA.
It builds on the SYCL specification from The `Khronos Group`_.
It is designed to allow developers to reuse code across hardware targets (CPUs and accelerators such as GPUs and FPGAs) and perform custom tuning for a specific accelerator.
ICX adopts `LLVM`_ for faster build times, and benefits from supporting the latest C++ standards.

NVCC
----
The `Nvidia CUDA Compiler (NVCC)`_ is a proprietary compiler by `NVIDIA`_ intended for use with `CUDA`_.
CUDA code runs on both the CPU and GPU.
NVCC separates these two parts and sends host code (the part of code which will be run on the CPU) to a C compiler like `GCC`_ or `Intel C++ Compiler Classic (ICC)`_ or `Microsoft Visual C++ Compiler`_, and sends the device code (the part which will run on the GPU) to the GPU.
The device code is further compiled by NVCC.
Like `ICX`_, NVCC is also based on `LLVM`_.

**Round Up**

If you have any questions, please feel free to reach out on `discord`_ in the `related work channel`_ or in the `related work forum`_!