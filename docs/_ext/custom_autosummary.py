from pathlib import Path

from docutils.parsers.rst import directives
from docutils import nodes
from docutils.statemachine import ViewList

import sphinx
from sphinx.ext.autosummary import Autosummary
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.errors import ExtensionError

logger = logging.getLogger(__name__)


class CustomAutosummary(Autosummary):
    new_option_spec = {
        "hide-table": directives.flag,
        "include": directives.flag,
        "fix-directory": directives.flag,
        "substitute-caption": directives.flag,
    }

    option_spec = {
        **Autosummary.option_spec,
        **new_option_spec,
    }

    def run(self):
        if all([option not in self.options for option in self.new_option_spec.keys()]):
            return super().run()

        # Documents which are generated then included using the 'include' option
        # have a messed up directory name.
        if "fix-directory" in self.options:
            self.check_for_prequisite("fix-directory", "toctree")

            candidates = [Path(x) for x in self.env.found_docs]
            candidates = set(
                [
                    x.parent
                    for x in candidates
                    if len(x.parts) > 1 and x.parts[-2] == self.options["toctree"]
                ]
            )

            current_doc = Path(self.env.docname).parent

            if len(candidates) == 1:
                self.options["toctree"] = str(candidates.pop().relative_to(current_doc))
            else:
                logger.warning(
                    "Could not find a single candidate for "
                    + f"{self.options.get('toctree')} while fixing toctree path.\n"
                    + f"Found {candidates}."
                )

        return_nodes = super().run()
        if "hide-table" in self.options:
            self.check_for_prequisite("hide-table", "toctree")
            # Auto summary produces some tables, and a toc tree at the end.
            # We only need the latter.
            return_nodes = return_nodes[-1:]

        if "include" in self.options:
            # Instead of adding a toctree, we include the newly generated files.
            self.check_for_prequisite("include", "toctree")

            # Unwrap the toctree node.
            toctree_node = return_nodes[-1]
            entries = [f"{name}.rst" for _, name in toctree_node[0]["entries"]]

            # Create custom RST
            rst = ViewList()

            for entry in entries:
                relative_entry = Path(entry).relative_to(Path(self.env.docname).parent)
                rst.append("", "fake.rst", offset=0)
                rst.append(f".. include:: {relative_entry}", "fake.rst", offset=1)
                rst.append("   :start-after: REMOVE_BEFORE_HERE", "fake.rst", offset=2)
                rst.append("", "fake.rst", offset=3)

            # Parse the RST
            node = nodes.section(ids=[f"autosummary-{entry}"])
            self.state.nested_parse(rst, self.content_offset, node)

            # Return the parsed RST
            # We need to return the toctree node as well, as it is used to
            # generate the stubs.
            return_nodes = [node]

        if "substitute-caption" in self.options:
            self.check_for_prequisite("substitute-caption", "include", included=False)
            self.check_for_prequisite("substitute-caption", "caption")
            self.check_for_prequisite("substitute-caption", "toctree")

            # We need to replace the caption of the toctree node.
            return_nodes[-1][0]["caption"] = self.config["ivy_toctree_caption_map"].get(
                return_nodes[-1][0]["caption"]
            )

        return return_nodes

    def check_for_prequisite(self, option, prequisite, included=True):
        if included and prequisite not in self.options:
            raise ExtensionError(
                f"'{option}' option is only valid when using '{prequisite}' option."
            )
        elif not included and prequisite in self.options:
            raise ExtensionError(
                f"'{option}' option is not valid when using '{prequisite}' option."
            )


def setup(app: Sphinx):
    app.setup_extension("sphinx.ext.autosummary")
    app.add_directive("autosummary", CustomAutosummary, override=True)
    app.add_config_value("ivy_toctree_caption_map", {}, True)

    return {
        "version": sphinx.__display_version__,
        "parallel_read_safe": True,
    }
