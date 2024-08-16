.. _`RWorks Ivy vs ONNX`:

Comparing Ivy with ONNX
=======================

.. _`Open Neural Network Exchange (ONNX)`: https://onnx.ai/
.. _`ONNX`: https://onnx.ai/

.. |onnx| image:: https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/related_work/exchange_formats/onnx.png
    :height: 20pt
    :class: dark-light
.. |ivy| image:: https://raw.githubusercontent.com/ivy-llc/ivy-llc.github.io/main/src/assets/full_logo_dark_long.svg#gh-dark-mode-only
    :height: 26pt
    :class: dark-light

ONNX |onnx|
-----------

Neural network exchange formats define a standardized file representation specifically for neural networks. The idea is that these can be used as
an intermediate representation for communicating or “exchanging” neural network architectures between different ML frameworks or between ML frameworks
and the target hardware. The focus is generally on simplifying the deployment of neural networks, with a typical workflow being: train the model,
save in an exchange format, and use this exchange format to communicate with the target compilers and hardware for model inference.

The `Open Neural Network Exchange (ONNX)`_ is a standardized static file format which fully defines the structure of a neural network and all of its weights.
Third parties can implement their own bindings to the ONNX standard format, which then enables the model to be saved to disk in the standard ONNX file format,
and be deployed on any hardware which supports the ONNX format. Some frameworks have also added support to "load in" ONNX models from disk, as well as support
for exporting to the format. This enables some degree of model conversion between frameworks, but generally only for model deployment and not training. ONNX
focuses on core neural network operations, with limited support for other more general array processing functions such as high order optimization,
signal processing, and advanced linear algebra.

|ivy|
-----------

The goal of Ivy is to be a comprehensive ML code conversion tool for all aspects of ML development, rather than solely focusing on deployment.
Ivy's transpiler uses a source code-to-source code approach to conversion, allowing any ML code to be converted. This includes models, functions, tools,
and entire libraries or codebases, providing a holistic solution for ML framework interoperability. Exchange formats like ONNX primarily work with deep
learning models, focusing on deployment and inference optimization. While ONNX offers a standardized file format that ensures consistent model representation
and efficient inference across diverse hardware platforms, it has limited support for model training and may not fully support advanced framework-specific
features. In contrast, Ivy supports both training and inference for converted models, enabling seamless end-to-end ML workflows within different frameworks.
By preserving the full functionality and flexibility of the original code, Ivy allows developers to leverage advanced features and optimizations specific to
each framework, thus addressing the limitations of exchange formats that only handle deep learning models for inference purposes. Ultmately, Ivy aims to solve
the broader challenge of enabling seamless development and flexibility across multiple ML frameworks, whereas ONNX addresses deployment of trained models
across various platforms and hardware environments.
