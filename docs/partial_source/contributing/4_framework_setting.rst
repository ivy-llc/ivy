Framework Setting
=================

.. _`this function`: https://github.com/unifyai/ivy/blob/0f131178be50ea08ec818c73078e6e4c88948ab3/ivy/framework_handler.py#L153

The backend framework is set by calling :code:`ivy.set_backend(backend_name)`. When calling `this function`_,
we go through the following steps:

#. store a global copy of the original :code:`ivy.__dict__` with no backend set.