youtube_map = {
    # TODO: Add video tutorials
    # "quickstart": "tmhFTFSEa6k",
}

nbsphinx_execute = 'never'
nbsphinx_prolog = """
.. |Open in Colab| raw:: html

    <a href="https://colab.research.google.com/github/unifyai/demos/blob/main/docs/{{ env.doc2path(env.docname, base=None) }}" target="_blank">
        <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>

.. |Github| raw:: html

    <a href="https://github.com/unifyai/demos/blob/main/docs/{{ env.doc2path(env.docname, base=None) }}" target="_blank">
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
