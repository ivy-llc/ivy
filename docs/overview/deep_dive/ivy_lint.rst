Ivy-Lint: Ivy's Custom Code Formatters
======================================

Overview
--------

``ivy-lint`` is a specialized suite of formatters crafted for the Ivy codebase. It addresses unique formatting requirements not catered to by standard Python formatters. While the suite currently highlights the ``FunctionOrderingFormatter``, we're continually expanding to include more formatters tailored to Ivy's needs.

Existing Formatters
-------------------

FunctionOrderingFormatter
~~~~~~~~~~~~~~~~~~~~~~~~~

This formatter ensures a standardized order of declarations within Python files, organizing functions, classes, and assignments based on a hierarchy designed for the Ivy codebase.

**Purpose**: To bring a sense of uniformity and structure to the code files by sorting various Python declarations.

**Target Files**: Specifically designed for frontends and tests.

How the Formatter Works:
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Header Management**:
   - Removes pre-existing headers in the source code based on specific patterns.

2. **Comments Handling**:
   - Extracts code components along with their leading comments, ensuring that relevant comments are retained during the reordering process.

3. **Dependency Handling**:
   - Constructs dependency graphs to understand and maintain the relationships between classes and assignments.

4. **Sorting Logic**:
   - Prioritizes imports, followed by assignments based on certain dependencies, then classes, and finally functions.
   - Preserves module-level docstrings at the top of the file.
   - Organizes helper functions and primary functions into separate sections for clarity.

5. **File Processing**:
   - Processes files that align with certain patterns, rearranging their content as needed.

Integration and Usage
---------------------

To get the best out of ``ivy-lint``, integrate it within a pre-commit hook. This ensures that whenever code changes are about to be committed, the suite checks and, if needed, formats the files to align with Ivy's standards.

For comprehensive details on weaving ``ivy-lint`` into your development practices, kindly refer to our `formatting guide <formatting.rst>`_.

Contribution
------------

We’re always thrilled to welcome contributions to ``ivy-lint``. If you're brimming with ideas for a new formatter or can enhance our existing ones, please connect with us either on our GitHub repository or our `discord <https://discord.gg/Y3prZYHS>`_ channel.

Round Up
--------

``ivy-lint`` stands as a testament to Ivy's commitment to code clarity and uniformity. As the landscape of our needs shifts, we foresee further refining and expanding our suite of formatters.

For all discussions or inquiries, you're always welcome on `discord <https://discord.gg/Y3prZYHS>`_ in the `formatting thread <https://discord.com/channels/799879767196958751/1190247322626572408>`_.
