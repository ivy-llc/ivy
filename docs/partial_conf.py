# Removing the Discord discussion channels for Ivy Framework
discussion_channel_map = {}

# Only generate docs for index.rst
# That resolved a bug of autosummary generating docs for code-block examples
# of autosummary
autosummary_generate = ["index.rst"]

skippable_method_attributes = [{"__qualname__": "_wrap_function.<locals>.new_function"}]

autosectionlabel_prefix_document = True

# Retrieve html_theme_options from docs/conf.py
from typing import List
from docs.conf import html_theme_options

html_theme_options["navbar_end"] = ["theme-switcher", "navbar-icon-links"]
html_theme_options.pop("switcher", None)
html_sidebars = {"**": ["custom-toc-tree"]}

repo_name = "ivy"

# Retrieve demos specific configuration
from docs.demos.demos_conf import *  # noqa

# Removing kapa.ai integration
html_js_files: List[str] = []
