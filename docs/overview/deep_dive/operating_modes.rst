Operating Modes
===============

.. _`array_significant_figures`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/__init__.py#L865
.. _`array_decimal_values`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/__init__.py#L904
.. _`warning_level`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/__init__.py#L931
.. _`nan_policy`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/__init__.py#L964
.. _`dynamic_backend`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/__init__.py#L998
.. _`precise_mode`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L87
.. _`array_mode`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L437
.. _`nestable_mode`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L490
.. _`exception_trace_mode`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L542
.. _`show_func_wrapper_trace_mode`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L597
.. _`min_denominator`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L2119
.. _`min_base`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L2174
.. _`queue_timeout`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L2444
.. _`tmp_dir`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L2502
.. _`shape_array_mode`: https://github.com/unifyai/ivy/blob/59cd7b5c4e2ca2fc6fc3c3ff728c3f210d9f740c/ivy/functional/ivy/general.py#L3418

Global Parameter Properties
---------------------------

There are a variety of global settings in ivy, each of which comes with: ``ivy.<setting>`` (getter), ``ivy.set_<setting>`` (setter), and ``ivy.unset_<setting>`` (unsetter).
Some of them are:

#. `array_significant_figures`_: Determines the number of significant figures to be shown when printing.
#. `array_decimal_values`_: Determines the number of decimal values to be shown when printing.
#. `warning_level`_: Determines the warning level to be shown when one occurs.
#. `nan_policy`_: Determines the policy of handling related to ``nan``.
#. `dynamic_backend`_: Determines if the global dynamic backend setting is active or not.
#. `precise_mode`_: Determines whether to use a promotion table that avoids any precision loss or a compute efficient table that avoids most wider-than-necessary promotions.
#. `array_mode`_: Determines the mode of whether to convert inputs to ``ivy.NativeArray``, then convert the outputs back to ``ivy.Array``.
#. `nestable_mode`_: Determines the mode of whether to check if function inputs are ``ivy.Container``.
#. `exception_trace_mode`_: Determines how much details of the ivy exception traces to be shown in the log.
#. `show_func_wrapper_trace_mode`_: Determines whether to show ``func_wrapper`` related traces in the log.
#. `min_denominator`_: Determines the global global minimum denominator used by ivy for numerically stable division.
#. `min_base`_: Determines the global global minimum base used by ivy for numerically stablestable power raising.
#. `queue_timeout`_: Determines the timeout value (in seconds) for the global queue.
#. `tmp_dir`_: Determines the name for the temporary folder if it is used.
#. `shape_array_mode`_: Determines whether to return shape as ``ivy.Array``.

Let's look into more details about getter and setter below!

Getter: ``ivy.<setting>`` attribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ivy.<setting>`` is a read-only static attribute. It acts as a getter and it will change internally whenever its related setter is used.

Should a user attempts to set the attribute directly, an error will be raised, suggesting them to change its value through the respective setter or unsetter.

.. code-block:: python
    >>> ivy.array_mode
    True
    >>> ivy.array_mode = False
    File "<stdin>", line 1, in <module>
    File ".../ivy/ivy/__init__.py", line 1306, in __setattr__
        raise ivy.utils.exceptions.IvyException(

    IvyException: Property: array_mode is read only! Please use the setter: set_array_mode() for setting its value!

Setter: ``ivy.set_<setting>`` and ``ivy.unset_<setting>`` functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to change the value of a property, setter functions must be used.

.. code-block:: python

    >>> ivy.array_mode
    True
    >>> ivy.set_array_mode(False)
    >>> ivy.array_mode
    False
    >>> ivy.unset_array_mode()
    >>> ivy.array_mode
    True
