.. _`RWorks What does Ivy Add?`:

What does Ivy Add?
==================

.. _`Array API Standard`: https://data-apis.org/array-api
.. _`EagerPy`: https://eagerpy.jonasrauber.de/
.. _`TensorLy`: http://tensorly.org/
.. _`Thinc`: https://thinc.ai/
.. _`NeuroPod`: https://neuropod.ai/
.. _`Keras`: https://keras.io/
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`torch.fx`: https://pytorch.org/docs/stable/fx.html
.. _`ONNX`: https://onnx.ai/
.. _`PyTorch`: https://pytorch.org/
.. _`JAX`: https://jax.readthedocs.io/
.. _`MLIR`: https://mlir.llvm.org/
.. _`Quansight`: https://quansight.com/
.. _`OctoML`: https://octoml.ai/
.. _`Modular`: https://www.modular.com/
.. _`Apache TVM`: https://tvm.apache.org/
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`related work channel`: https://discord.com/channels/799879767196958751/1034436036371157083
.. _`related work forum`: https://discord.com/channels/799879767196958751/1034436085587120149
.. _`Flux`: https://fluxml.ai/
.. _`Julia`: https://julialang.org/

API Standards
-------------
Ivy fully adheres to the `Array API Standard`_, and we are strongly aligned with their unification vision.
Ivy is entirely complimentary to the standard, implementing all of the functions defined by the standard, and populating each of them with implementations supporting a variety of frameworks.

Further into the future, we have quite lofty ambitions for Ivy to extend beyond the Python realm, and to help build the bedrock of a more general, language-agnostic, purely mathematical standardized Array API.
Given that the function signatures at the Array API level are very much mathematically bound, it is certainly possible to define this purely mathematical, language-agnostic Array API, using the Array API Standard as a starting point.
In this light, the current release of Ivy would simply be the "Python backend" of Ivy, which itself of course would contain the same framework-specific backends itself, but just pushed one level further down the backend hierarchy.
In future, we hope that it will be possible to use this mathematical API as an intermediate representation which would enable transpilations between frameworks in *different* languages.
For example, transpiling from `PyTorch`_ to the `Flux`_ framework written in `Julia`_ would be a great feature.
However, for the time being we are focusing exclusively on Python, in order to mitigate the risk of "biting off more than we can chew"!

Wrapper Frameworks
------------------
Ivy is itself a Python Wrapper Framework.
The biggest difference between Ivy and all others listed in the :ref:`Wrapper Frameworks` section is that Ivy supports transpilations between frameworks, while all other frameworks only enable the creation of entirely new code which itself is framework-agnostic.
There are also other more subtle differences.
For example, Ivy includes both a low level fully functional API and a high level stateful API, offering both low level control and high level convenience.
In contrast, `EagerPy`_ and `TensorLy`_ both only include functional APIs, `Thinc`_ only includes a high level stateful API, and `NeuroPod`_ only supports an even higher level wrapper for deployment.
Similar to Ivy, `Keras`_ did also support both functional and stateful APIs, but since version 2.4 it only supports `TensorFlow`_ as a backend.

Frameworks
----------
Ivy wraps the standalone ML frameworks in Python, and enables transpilations between the various frameworks and framework versions.
It therefore extends what is possible in any of the specific individual frameworks in isolation.

Graph Tracers
-------------
Ivy’s :ref:`Graph Compiler` exhibits similar properties to many of the framework-specific graph tracers.
Ivy’s graph compiler employs function tracing for computing the graph, and uses this graph as an intermediate representation during the transpilation process.
Of all the graph tracers, Ivy’s graph compiler is most similar to `torch.fx`_.
This is because :code:`torch.fx` also operates entirely in Python, without deferring to lower level languages for tracing and extracting the computation graph or the intermediate representation.
The main difference is that Ivy’s graph compiler is fully framework-agnostic; Ivy’s compiler is able to compile graphs from any framework, while framework-specific compilers are of course bound to their particular framework.

Exchange Formats
----------------
The neural network exchange formats have particular relevance to Ivy, given their utility for sharing networks between frameworks.
For example, models can be exported to the `ONNX`_ format from `TensorFlow`_, and then the ONNX model can be loaded into `PyTorch`_.
This could be seen as a form of transpilation, which is one of the central goals of Ivy.
However, there are some important differences between Ivy’s approach and that of exchange formats.
Firstly, Ivy requires no third-party "buy in" whatsoever.
We take the initiative of wrapping the functional APIs of each framework and each framework version ourselves, such that Ivy can transpile between any framework and version without any need for the existing frameworks to put their own time and effort into supporting Ivy.
Additionally, Ivy also enables transpilations for training, not only for deployment.
For example, Ivy enables users to fine-tune or retrain `JAX`_ models using their `PyTorch`_ training pipeline, something which the exchange formats do not enable.
Finally, Ivy exhaustively supports the full range of array processing functions available in all frameworks, which again the exchange formats do not do.
This makes it much more broadly applicable to a wider range of applications, spanning from cutting edge deep learning to more conventional machine learning, general numerical computing and data analytics.

Compiler Infrastructure
-----------------------
Compiler infrastructure is essential in order to enable arbitrary frameworks to support arbitrary hardware targets.
`MLIR`_, for example, has hugely simplified `TensorFlow`_'s workflow for supporting various hardware vendors in a scalable manner, with minimal code duplication.
However, while infrastructure such as MLIR at the compiler level is essential for framework developers, in its current form it cannot easily be used to guide the creation of tools which enable code transpilations between the user facing functions higher up the ML stack.
The intermediate representations used by the compiler infrastructure sit further down the stack, closer to the compilers and to the hardware.
Transpilation between frameworks requires an IR that sits between the functional APIs of the frameworks themselves, in the way that Ivy does, and this is not really the purpose of these compiler infrastructure IRs.
Ivy's unification goals are therefore complimentary to the unifying goals of the various compiler infrastructure efforts.

Multi-Vendor Compiler Frameworks
--------------------------------
Multi-vendor compiler frameworks sit a bit further down the stack still, and can optionally make use of the compiler infrastructure as scaffolding.
Likewise, these greatly simplify the complex task of enabling models from any framework to be deployed on any hardware, but they do nothing to address the challenge of running code from one framework inside another framework at training time, which is the central problem Ivy addresses.
Therefore, again these efforts are complimentary to Ivy's high-level unification goals.

Vendor-Specific APIs
--------------------
Likewise, vendor-specific APIs sit even further down the stack.
These enable custom operations to be defined for execution on the specified hardware, and they form an essential part of the stack.
However, again they do nothing to address the challenge of running code from one framework inside another framework at training time, which is the central problem Ivy addresses.

Vendor-Specific Compilers
-------------------------
Finally, vendor-specific compilers sit at the very bottom of the stack as far as our diagram is concerned (ignoring assembly languages, byte code etc.).
These are essential for converting models into instructions which the specific hardware can actually understand, and they also of course form a critical part of the stack.
However, again they do nothing to address the challenge of running code from one framework inside another framework at training time, which is the central problem Ivy addresses.

ML-Unifying Companies
---------------------
The ML-unifying companies `Quansight`_, `OctoML`_ and `Modular`_ are/were directly involved with the `Array API Standard`_, `Apache TVM`_ and `MLIR`_ respectively, as explained in the :ref:`ML-Unifying Companies` section.
For the same reasons that Ivy as a framework is complementary to these three frameworks, Ivy as a company is also complementary to these three companies.
Firstly, we are adhering to the `Array API Standard`_ defined by Quansight.
In essence they have written the standard and we have implemented it, which is pretty much as complementary as it gets.
Similarly, OctoML makes it easy for anyone to *deploy* their model anywhere, while Ivy makes it easy for anyone to mix and match any code from any frameworks and versions to *train* their model anywhere.
Again very complementary objectives.
Finally, Modular will perhaps make it possible for developers to make changes at various levels of the stack when creating ML models using their "", and this would also be a great addition to the field.
Compared to Modular which focuses on the lower levels of the stack, Ivy instead unifies the ML frameworks at the functional API level, enabling code conversions to and from the user-facing APIs themselves, without diving into any of the lower level details.
All of these features are entirely complementary, and together would form a powerful suite of unifying tools for ML practitioners.

**Round Up**

If you have any questions, please feel free to reach out on `discord`_ in the `related work channel`_ or in the `related work forum`_!