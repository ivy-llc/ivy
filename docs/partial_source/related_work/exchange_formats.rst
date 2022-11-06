Exchange Formats
================

.. _`Open Neural Network Exchange (ONNX)`: https://onnx.ai/
.. _`ONNX`: https://onnx.ai/
.. _`Neural Network Exchange Format (NNEF)`: https://www.khronos.org/nnef
.. _`CoreML`: https://developer.apple.com/documentation/coreml
.. _`Khronos Group`: https://www.khronos.org/
.. _`Apple`: https://www.apple.com/
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`related work channel`: https://discord.com/channels/799879767196958751/1034436036371157083
.. _`related work forum`: https://discord.com/channels/799879767196958751/1034436085587120149

Neural network exchange formats define a standardized file representation specifically for neural networks.
The idea is that these can be used as an intermediate representation for communicating or “exchanging” neural network architectures between different ML frameworks or between ML frameworks and the target hardware.
The focus is generally on simplifying deployment of neural networks, with a typical workflow being: train the model, save in an exchange format, use this exchange format to communicate with the target compilers and hardware for model inference.

ONNX
----
The `Open Neural Network Exchange (ONNX)`_ is a standardized static file format which fully defines the structure of a neural network and all of its weights.
Third parties can implement their own bindings to the ONNX standard format, which then enables the model to be saved to disk in the standard ONNX file format, and be deployed on any hardware which supports the ONNX format.
Some frameworks have also added support to "load in" ONNX models from disk, as well as support for exporting to the format.
This enables some degree of model conversion between frameworks, but generally only for model deployment and not training.
ONNX focuses on core neural network operations, with limited support for other more general array processing functions such as high order optimization, signal processing, and advanced linear algebra.

NNEF
----
Similar to `ONNX`_, the `Neural Network Exchange Format (NNEF)`_ is also a standardized static file format which fully defines the structure of a neural network and all of its weights, with some support also for training tools.
The format was developed and is maintained by the `Khronos Group`_.
Overall, NNEF shares a lot of similarities with ONNX, but has not reached the same level of adoption.

CoreML
------
`CoreML`_ itself is not an exchange format, it is a framework which enables models to be trained and deployed on `Apple`_ devices with a simple zero-code interactive interface.
However, CoreML is built upon its own Core ML format, and Apple have open sourced :code:`coremltools`, which provides a set of tools to convert ML models from various frameworks into the Core ML format.
The Core ML format is itself an exchange format, albeit with the sole purpose of exchanging to Apple’s CoreML framework, rather than enabling exchanges between multiple different parties as is the case for the other exchange formats.

**Round Up**

If you have any questions, please feel free to reach out on `discord`_ in the `related work channel`_ or in the `related work forum`_!