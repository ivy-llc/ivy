FAQ
===

.. _`dex`: https://github.com/dexidp/dex
.. _`API for distributed training`: https://github.com/unifyai/ivy/blob/a2f37b1bae232b7ba5257e59f8b46a0374cca9f1/ivy/functional/ivy/device.py#L660
.. _`fully support these`: https://pytorch.org/tutorials/prototype/vmap_recipe.html
.. _`Ivy Builder`: https://github.com/unifyai/builder
.. _`README`: https://github.com/unifyai/ivy

These are some of the most common technical questions that continue to arise when we're discussing Ivy with developers
in the community.

As Ivy becomes more mature and we continue these discussions,
then many more questions and answers will no doubt be added!

We are all incredibly grateful to everyone in the community who has put in the time and effort to ask so many important
and probing questions! We hope these Q&As are a useful reference!

Maintaining Backend Versions
----------------------------

**Q:** Isn't it complex to maintain support for all backend versions,
particularly as they undergo constant changes?
How are you going to handle this,
will you have an option to select any version for any backend?

**A:** Ivy **only** wraps the functional APIs of each backend framework.
The last 2 years of Ivy development have shown us how remarkably stable the functional
APIs are for each backend framework.
**Not once** have we needed to change an implementation or a unit test as a result of a
version update of a backend framework. This is not entirely surprising,
each framework has strong backward compatibility requirements,
and the functional API is generally one of the lower level building blocks upon
which everything else in the framework depends.
Our CI always tests against the latest version available on PyPI,
and this has been the case since we started development.
We do not lock-in any versions during our continuous testing,
and we will continue to always pull the latest version.

In future, we hope to add explicit testing also for previous versions,
so we can guarantee backward compatibility for each backend.
We will also add an option to select backend versions for the small minority of cases
where changes in the backend functional APIs do cause breaking changes for Ivy.

Dynamic Sizes
-------------

**Q:** Assuming a static computation graph, can tensors have sizes that dynamically change?
XLA does not support dynamic sizes, because it JIT-compiles the graph, and pre-allocates all buffers in memory before
the graph runs. TensorFlow and PyTorch do allow dynamic sizes, but only on certain backends.
Dynamic sizes require a dynamic memory memory manager, which CPUs/GPUs have, but XLA currently doesn't.
How does Ivy deal with all of this?

**A:** Ivy assumes dynamic shapes are supported, but an error will be thrown if/when the function is compiled
with dynamic shapes enabled, but the backend does not support dynamic shapes in the compiled graph.
For now, fully framework-agnostic compiled graphs are only possible for static graphs.

Type and Shape Checking
-----------------------

**Q:** What kind of type system does Ivy use?  Does it do shape-checking of tensors? If so, how does it handle dynamic
sizes? The gold standard here is a fully dependent type system, but this is very rare, with the exception of `dex`_.

**A:**  The checks performed during graph compilation will remain backend-specific. The function :func:`ivy.compile`
wraps the backend compilation functions, for example :func:`jax.jit`, :func:`tf.function`, :func:`torch.jit.script` and
:func:`torch.jit.trace`. For some backends, shape-checking will be performed during the compilation phase and for others
it will not.

GPU handling
------------
**Q:** How does Ivy handle GPU usage? 

**A:** Ivy handles GPU usage by simply wrapping the backend frameworks, and so Ivy will use GPUs in the same manner as the 
backend framework does. E.g. When using a torch backend, then torch will be a dependency of Ivy, and its handling of GPU 
functionalities will be inherited and extended upon by Ivy.

Model Deployment
----------------
**Q:** Does Ivy support model deployment?
**A:** Yes, Ivy will support efficient model deployment. However, currently this feature is not yet supported as the graph compiler 
module is still under development, and will be released soon with ivy version 1.2.0.


Dynamic Control Flow
--------------------
**Q:** Tensorflow has dynamic control-flow primitives (loops, branches) even within a static computation graph.
Jax also has dynamic control-flow (:code:`lax.scan`, :code:`lax.while`), but support is limited; only :code:`lax.scan`
is differentiable in reverse mode.

Branching is also tricky, and is backend-dependent. CPUs have branch predictors and can execute tight loops, GPUs don't,
but have drivers that can schedule kernels dynamically, some other architectures do static scheduling,
which limits the kinds of algorithms that can run effectively.

TensorFlow eager and PyTorch allow you to use full python control flow, (loops, branches, function calls,
dynamic dispatch, recursion) but there is no static computation graph. How will Ivy handle dynamic control flow?
Will Ivy parse python ASTs?

**A:** For now, Ivy will not support dynamic control flow by parsing ASTs. The dynamism of :code:`for` loops and
:code:`while` loops will be ignored during compilation, and just the static trace which chains the array operations
performed during the forward pass at compile time will be preserved.

However, Ivy will support the compilation of looping and branching methods such as :code:`lax.scan`, :code:`lax.while`,
:code:`tf.while`, :code:`tf.cond` etc.
In cases where there is not an associated compilable method in other backends,
we will strive to implement this as a composition of existing compilable operations.
If such a composition is not possible, then we will instead convert these to compositions of pure Python :code:`for`,
:code:`while` and :code:`if` statements (when using a PyTorch backend for example).

The reverse mode conversions will not be possible without using parsing ASTs though.
This does mean that for example TensorFlow (with loops + branches) → PyTorch (with for, while + if statements)
but the reverse mode will not preserve the loops and branches PyTorch (with for, while + if statements) → TensorFlow (static, no loops or branches).

Auto-Differentiation
--------------------

**Q:** How do you handle reverse mode, forward mode, and Jacobians?  How about stop gradients, and gradient
checkpointing, and custom gradients? What about autodiff for control-flow operators like :code:`lax.scan`?
This is where JAX really shines, and unless you are implementing your own autodiff framework, you are at the mercy of
whatever the backend supports.

**A:** Ivy will implement all of the general methods that JAX supports, and will provide errors if/when the backend
does not support this. In general, Ivy will support the superset of functionality, and not just the lowest common
denominator. Ivy takes a fully functional approach like JAX, and the API enables arbitrary nested
:code:`execute_with_gradient` calls up to an arbitrary gradient order. Again, if a backend does not support this then an
error will be thrown. This means Ivy code is not 100% framework-agnostic, and is indeed at the mercy of what the backend
autograd package supports in these cases.

Replicas, and Data vs Model Parallelism
---------------------------------------

**Q:** Big models don't run on just one device, and the major frameworks have *very* different ways of splitting a model
up so that it runs on a cluster. There are multiple competing paradigms for parallelisation -- e.g. SPMD vs mixture of
experts. JAX and Tensorflow are very sophisticated in this department, and routinely run models on hundreds or
thousands of devices. How will Ivy support multi-device training, if at all?

**A:** This is not something we’re diving into too deeply at the moment. However, we have written our own `API
for distributed training`_, which broadly follows PyTorch’s approach using a CUDA-enabled multiprocessing module.

If heavily distributed training is important. Then Ivy can be supplementary for the time being, rather than a total
replacement. For example, someone can use TensorFlow’s distributed training tools, and just use
Ivy to copy over a PyTorch model into their TF pipeline.

We are not trying to encourage anyone to drop any existing tools and just use Ivy instead.
Projects can use 1% Ivy code or 100%. We’re very happy in either case!

Support for Functions
---------------------

**Q:** Is it possible to compile tensor code into a reusable and differentiable function?  If you can't, then it will
be difficult to apply any fancy kernel fusion algorithms, and you can expect to lose a lot of performance.
What about higher-order operations, like :code:`jax.vmap` and :code:`jax.pmap`?

**A:** Most functions in Ivy are *primary* functions, which are generally implemented as light wrapping around a
near-identical backend-specific function, which itself will likely map to an efficient kernel. *Compositional* functions
on the other hand are implemented as a composition of other Ivy functions, meaning there will not be a one-to-one
mapping to a single backend kernel. However, our experiments (to be published soon!) show this does not lead to a
significant run-time overhead, even when a composition of operations is required.

For methods like :code:`jax.vmap` and :code:`jax.pmap`, we will need to implement these as (possibly inefficient)
compositions in other frameworks, until they are supported in these frameworks. However, it seems as though other
frameworks such as PyTorch are seeing the benefit in these functions, and will eventually `fully support these`_.

Alternative Data Structures
---------------------------

**Q:** Will Ivy support data structures such as tuples, dictionaries, lists etc.? For example, JAX code is full of them.

**A:** We will of course support these structures in pure python code, but we will not support backend-specific
alternative compilable data structures. While Ivy will not provide an interface to these data structures directly,
Ivy code can easily supplement JAX code which does contain these data structures,
and both can be compiled together without issue. Ivy can act as a supplementary framework if/when some of the more
unique backend-specific data structures are required.

Custom Operations
-----------------

**Q:** Most frameworks have a backdoor for user-defined ops, implemented in C++/CUDA, or some kind of host callback
mechanism. Will Ivy support this ability also?

**A:** We will not attempt to provide a unified back-door for all possible backend kernel customizations,
but of course users can still use the backend-specific backdoors which already exist when using Ivy.

The Pipeline
------------

**Q:** How will Ivy manage the training loop and input pipeline?  What about loading and saving models, recording of
scalar metrics, visualization, etc.? These are often also somewhat framework-dependent.

**A:** We are not advocating to replace all code with Ivy. We would encourage users to continue using
whatever data loaders they want to, and perhaps just use an Ivy model, or use Ivy to convert a model, or even just a
single function from a library. If users want to use Ivy more deeply, then they can use `Ivy Builder`_,
which includes framework-agnostic abstract data loaders, trainers, and other higher level classes for composing full
training pipelines.

State
-----

**Q:** Tensorflow handles state as part of the static graph. JAX is purely functional and so outsources it to one of
several third-party libraries, like Flax. How will Ivy handle state?

**A:** Ivy has a fully functional backend. When using a TensorFlow or PyTorch backend, we pass all of the variables and
gradients explicitly as function inputs and outputs. This is not actually required for the stateful back-ends, but we
still return the values such that JAX is also supported. Ivy will remain fully functional in design, and we therefore
assume behavior similar to JAX. Our simple example on the `README`_ trains correctly for all back-ends, which passes
everything explicitly in a functional manner.
