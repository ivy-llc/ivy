.. _`RWorks API Standards`:

API Standards
=============

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`discord`: https://discord.gg/sXyFF8tDtm

API standards are standardized application programming interfaces (APIs) which define the function signatures which similar libraries should adhere to for maximal interoperability between those libraries.

Array API Standard
------------------

The `Array API Standard`_ defines a unified application programming interface (API) for Python libraries which perform numerical operations on high dimensional arrays (tensors).
This standard can be considered as “higher level” than the ML frameworks themselves, given that the standard defines the functions without implementing them, whereas the frameworks include implementations for all of the functions which fit into this standard API, with all the lower level considerations also handled within these implementations.
The Array API Standard takes a lowest common denominator approach, whereby each function in the standard represents the minimum behaviors of the function without restricting extensions to the function.
This means that two very different libraries can adhere to the same standard, despite having very different extended behaviors for some of the functions in the standard.
The standard is also not exhaustive.
For example, there are <insert_number> functions defined in the standard, whereas the functions defined in each framework are as follows:

<insert_table>

Therefore, two frameworks which adhere to the standard will still have major differences by virtue of the extra functions they support which are not present in the standard.
