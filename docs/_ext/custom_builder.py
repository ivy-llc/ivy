from docutils import nodes

from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.util import logging

logger = logging.getLogger(__name__)


class IvyHtmlBuilder(StandaloneHTMLBuilder):
    def write_doc(self, docname: str, doctree: nodes.document) -> None:
        if docname.startswith("docs/functional"):
            doctree.walk(
                ReplaceNodeVisitor(doctree, r"ivy\.functional\.ivy\..*\.", "ivy.")
            )

        return super().write_doc(docname, doctree)


class ReplaceNodeVisitor(nodes.NodeVisitor):
    def __init__(self, document: nodes.document, pattern: str, replacement: str):
        super().__init__(document)
        self.pattern = pattern
        self.replacement = replacement

    def visit_Text(self, node: nodes.Text):
        if node.startswith("ivy.functional.ivy"):
            new_text = nodes.Text("ivy.")
            node.parent.replace(node, new_text)

    def unknown_visit(self, node: nodes.Node):
        pass


def setup(app: Sphinx):
    app.add_builder(IvyHtmlBuilder, override=True)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
    }
