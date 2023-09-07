Ivy-Lint: Ivy's Custom Code Formatters
======================================

Overview
--------

`ivy-lint` is a specialized suite of formatters designed to ensure that code in the Ivy codebase adheres to certain standards and organization. The formatters under `ivy-lint` address specific concerns not covered by other standard Python formatters. Although currently it mainly includes the `FunctionOrderingFormatter`, plans are in motion to expand this suite to cater to other code organizational needs.

Existing Formatters
-------------------

FunctionOrderingFormatter
~~~~~~~~~~~~~~~~~~~~~~~~~

The `FunctionOrderingFormatter` is designed to ensure a specific order of declarations in Python files. It sorts functions, classes, and assignments in the codebase according to a predefined hierarchy.

**Purpose**: The primary objective of this formatter is to impose order in the code files by sorting Python declarations.

**Target Files**: It targets specific files that match patterns defined in `FILE_PATTERN`.

The `FunctionOrderingFormatter` in `ivy-lint` is a specialized formatter designed to ensure a specific order of declarations in Python files. It sorts the functions, classes, and assignments in the codebase according to a predefined hierarchy.

How the Formatter Works:
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Header Removal**: 
    - Before any ordering is performed, the existing headers in the source code are removed using the regex pattern defined in `HEADER_PATTERN`.

2. **Extracting Node Comments**: 
    - The formatter extracts AST nodes along with their leading comments from the source code. Leading comments are those found right above a declaration or a statement.
    - The method `_extract_all_nodes_with_comments` provides this extraction, and `_extract_node_with_leading_comments` aids by retrieving a specific node's comments.

3. **Building Dependency Graphs**: 
    - To understand and maintain inherent relationships, dependency graphs are constructed.
    - **Class Dependencies**: A graph (`class_dependency_graph`) is built to understand class inheritance. Nodes represent class names, and directed edges represent inheritance. The `class_build_dependency_graph` method facilitates this.
    - **Assignment Dependencies**: Another graph (`assignment_dependency_graph`) captures dependencies among assignments. For instance, if one assignment depends on the value from another assignment, this relationship is represented in the graph. This is facilitated by the `assignment_build_dependency_graph` method.

4. **Sorting Logic**:
    - The `sort_key` function dictates the order in which nodes are arranged:
        1. Imports are prioritized.
        2. Assignments come next, with various considerations. If assignments depend on another assignment or a function/class, they are given priority.
        3. Classes follow, based on the inheritance chain.
        4. Functions come in two categories: Helper (private with names starting with "_") and API functions (public). They're sorted accordingly.
    - Any module-level docstring is preserved and positioned at the beginning.
    - Comments are retained with their associated code sections.
    - Helper functions are grouped under the `# --- Helpers --- #` header, while the primary functions come under `# --- Main --- #`.
   
5. **File Processing**:
    - If a file matches the `FILE_PATTERN`, the formatter reads its content and applies the rearrangement logic.
    - The reordered code is then written back to the file.
    - If there's a `SyntaxError` during the process, the formatter will notify that the provided file does not contain valid Python code.

Integration and Usage
---------------------

To utilize any formatter within `ivy-lint`, integrate it as part of a pre-commit hook. When the hook triggers (usually before committing changes to a repository), the formatters in `ivy-lint` check each file and, if necessary, process them to ensure they adhere to the desired standards.

For a step-by-step guide on integrating `ivy-lint` with your development workflow, refer to our comprehensive [pre-commit guide].

Contribution
------------

We welcome contributions to `ivy-lint`! If you have an idea for a new formatter or improvements for the existing ones, please raise an issue on our GitHub repository or discuss it with us on our `discord`_ channel.

Round Up
--------

`ivy-lint` is an essential tool in Ivy's arsenal to maintain code consistency and readability. As we move forward, we anticipate the addition of more specialized formatters to address Ivy's evolving needs.

For any queries or discussions, please join us on `discord`_ in the `formatting channel`_!
