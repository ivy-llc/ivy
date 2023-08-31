# Retrieve html_theme_options from docs/conf.py
from docs.conf import html_theme_options

html_theme_options["switcher"]["json_url"] = "https://unify.ai/docs/versions/ivy.json"

nbsphinx_execute = 'never'
nbsphinx_prolog = """
|Open in Colab| |Github|

.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/unifyai/demos/blob/main/docs/{{ env.doc2path(env.docname, base=None) }}

.. |Github| image:: https://badgen.net/badge/icon/github?icon=github&label
    :target: https://github.com/unifyai/demos/blob/main/docs/{{ env.doc2path(env.docname, base=None) }}
"""
