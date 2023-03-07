import re

from docutils import nodes

from sphinx.util.docutils import SphinxDirective
from sphinx.application import Sphinx


class DiscussionLinks(SphinxDirective):

    has_content = True

    def run(self):
        module = self.content[0]
        if module not in self.config["discussion_channel_map"]:
            return []

        text = self.config["discussion_paragraph"]

        text = text.replace("{{submodule}}", module.split(".")[-1])
        text = text.replace("{{discord_link}}", self.config["discord_link"])
        text = text.replace("{{channel_link}}", self.config["channel_link"])
        text = text.replace("{{forum_link}}", self.config["forum_link"])
        text = text.replace(
            "{{channel_id}}", self.config["discussion_channel_map"][module][0]
        )
        text = text.replace(
            "{{forum_id}}", self.config["discussion_channel_map"][module][1]
        )

        # Convert markdown style links to reference nodes
        paragraph_node = nodes.paragraph()
        plain_text_list = re.split(r"\[(.+?)\]\((.+?)\)", text)
        in_link = text[0] == "["

        while len(plain_text_list) > 0:
            text = plain_text_list.pop(0)
            if in_link:
                link_text = text
                link_url = plain_text_list.pop(0)
                paragraph_node += nodes.reference(link_text, link_text, refuri=link_url)
                in_link = False
            else:
                paragraph_node += nodes.Text(text)
                in_link = True

        return [paragraph_node]


def setup(app: Sphinx):
    app.add_directive("discussion-links", DiscussionLinks)
    # Add configurable paragraph
    app.add_config_value(
        "discussion_paragraph",
        "This should have hopefully given you an overview of the {{submodule}} "
        + "submodule, if you have any questions, please feel free to reach out on our "
        + "[discord]({{discord_link}}) in the [{{submodule}} channel]({{channel_link}})"
        + " or in the [{{submodule}} forum]({{forum_link}})!",
        "env",
    )
    app.add_config_value("discord_link", "https://discord.gg/ZVQdvbzNQJ", "env")
    app.add_config_value(
        "channel_link",
        "https://discord.com/channels/799879767196958751/{{channel_id}}",
        "env",
    )
    app.add_config_value(
        "forum_link",
        "https://discord.com/channels/799879767196958751/{{forum_id}}",
        "env",
    )
    app.add_config_value("discussion_channel_map", {}, "env")

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
