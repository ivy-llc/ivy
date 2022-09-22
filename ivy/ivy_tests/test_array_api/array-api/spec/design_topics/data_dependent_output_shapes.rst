.. _data-dependent-output-shapes:

Data-dependent output shapes
============================

Array libraries which build computation graphs commonly employ static analysis that relies upon known shapes. For example, JAX requires known array sizes when compiling code, in order to perform static memory allocation. Functions and operations which are value-dependent present difficulties for such libraries, as array sizes cannot be inferred ahead of time without also knowing the contents of the respective arrays.

While value-dependent functions and operations are not impossible to implement for array libraries which build computation graphs, this specification does not want to impose an undue burden on such libraries and permits omission of value-dependent operations. All other array libraries are expected, however, to implement the value-dependent operations included in this specification in order to be array specification compliant.

Value-dependent operations are demarcated in this specification using an admonition similar to the following:

.. admonition:: Data-dependent output shape
   :class: important

   The shape of the output array for this function/operation depends on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function/operation difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.
