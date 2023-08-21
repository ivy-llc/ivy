.. _`RWorks Graph Tracers`:

Graph Tracers
=============

.. _`TensorFlow`: https://tensorflow.org/
.. _`JAX`: https://jax.readthedocs.io/
.. _`PyTorch`: https://pytorch.org/
.. _`FX`: https://pytorch.org/docs/stable/fx.html
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`related work channel`: https://discord.com/channels/799879767196958751/1034436036371157083
.. _`related work forum`: https://discord.com/channels/799879767196958751/1034436085587120149

Graph tracers enable acyclic directed computation graphs to be extracted from functions which operate on the tensors, expressed as source code in the framework.
There is inevitably some overlap with the role of the lower level compilers here, but for the purpose of this discussion, we consider tracers as being any tool which executes the function to be traced and produces a computation graph consisting solely of the lowest level functions defined within the framework itself, without going any lower.
In this light, the tracer does not need to know about the hardware, the compiler instruction set, or anything else lower level.
It simply creates an acyclic directed graph which maps the inputs of a function to the outputs of a function, as a composition of the low level functions defined within the framework.
This is a very useful representation which can then make subsequent compilation simpler, and so this graph representation often sits between the raw source code and the lower level compilers which compile to specific hardware.

tf.Graph
--------
The :code:`tf.Graph` class represents an arbitrary `TensorFlow`_ computation, represented as a dataflow graph.
It is used by :code:`tf.function` to represent the function's computations.
Each graph contains a set of :code:`tf.Operation` instances, which represent units of computation; and :code:`tf.Tensor` instances, which represent the units of data that flow between operations.

Jaxpr
-----
Conceptually, one can think of `JAX`_ transformations as first trace-specializing the Python function to be transformed into a small and well-behaved intermediate form that is then interpreted with transformation-specific interpretation rules.
It uses the actual Python interpreter to do most of the heavy lifting to distill the essence of the computation into a simple statically-typed expression language with limited higher-order features.
That language is the jaxpr language.
A :code:`jax.core.Jaxpr` instance represents a function with one or more typed parameters (input variables) and one or more typed results.
The results depend only on the input variables; there are no free variables captured from enclosing scopes.

torch.jit
---------
:code:`torch.jit.trace` and :code:`torch.jit.trace_module` enables a module or Python function to be traced in `PyTorch`_, and an executable is returned which will be optimized using just-in-time compilation.
Example inputs must be provided, and then the function is run, with a recording of the operations performed on all the tensors.
The resulting recording of a standalone function produces a :code:`ScriptFunction` instance.
The resulting recording of :code:`nn.Module.forward` or :code:`nn.Module` produces a :code:`ScriptModule` instance.
This module also contains any parameters that the original module had as well.

torch.fx
--------
`FX`_ is a toolkit for developers to use to transform :code:`torch.nn.Module` instances in `PyTorch`_.
FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation.
The symbolic tracer performs “symbolic execution” of the Python code.
It feeds fake values, called Proxies, through the code.
Operations on these Proxies are recorded.
The intermediate representation is the container for the operations that were recorded during symbolic tracing.
It consists of a list of Nodes that represent function inputs, call-sites (to functions, methods, or :code:`torch.nn.Module` instances), and return values.
The IR is the format in which transformations are applied.
Python code generation is what makes FX a Python-to-Python (or Module-to-Module) transformation toolkit.
For each Graph IR, valid Python code matching the Graph’s semantics can be created.
This functionality is wrapped up in GraphModule, which is a :code:`torch.nn.Module` instance that holds a Graph as well as a forward method generated from the Graph.

Taken together, this pipeline of components (symbolic tracing -> intermediate representation -> transforms -> Python code generation) constitutes the Python-to-Python transformation pipeline of FX.
In addition, these components can be used separately.
For example, symbolic tracing can be used in isolation to capture a form of the code for analysis (and not transformation) purposes.
Code generation can be used for programmatically generating models, for example from a config file.

**Round Up**

If you have any questions, please feel free to reach out on `discord`_ in the `related work channel`_ or in the `related work forum`_!
