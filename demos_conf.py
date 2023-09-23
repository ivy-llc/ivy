youtube_map = {
    "demos/quickstart": "9myf5tekBWU",
    "demos/learn_the_basics/01_write_ivy_code": "lYW_V4ZkYEo",
    "demos/learn_the_basics/02_unify_code": "YKrxYB-1Xio",
    "demos/learn_the_basics/03_compile_code": "fqnhEi-qA4w",
    "demos/learn_the_basics/04_transpile_code": "6WCYOF1vdLw",
    "demos/learn_the_basics/05_lazy_vs_eager": "xav4HYfvPOU",
    "demos/learn_the_basics/06_how_to_use_decorators": "fueduIEMkbY",
}

nbsphinx_execute = 'never'
nbsphinx_prolog = """
.. |Open in Colab| raw:: html

    <a href="https://colab.research.google.com/github/unifyai/demos/blob/main/{{ env.doc2path(env.docname, base=None)|replace("demos/", "", 1) }}" target="_blank">
        <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>

.. |Github| raw:: html

    <a href="https://github.com/unifyai/demos/blob/main/{{ env.doc2path(env.docname, base=None)|replace("demos/", "", 1) }}" target="_blank">
        <img src="https://badgen.net/badge/icon/github?icon=github&label">
    </a>

{% if env.config.youtube_map[env.docname] %}
.. raw:: html

    <h4 style="margin-top: .05rem;">Video Tutorial</h4>

.. raw:: html

    <iframe width="560" height="315" style="margin-bottom: 1rem;"
        src="https://www.youtube.com/embed/{{ env.config.youtube_map[env.docname] }}"
        frameborder="0" allow="encrypted-media; picture-in-picture" allowfullscreen>
    </iframe>
    <br>
{% endif %}

|Open in Colab| |Github|
"""


def setup(app):
    app.add_config_value("youtube_map", youtube_map, "env")
