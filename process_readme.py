import re


def preprocess_readme(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Remove img tags that have class "only-dark"
    content = re.sub(
        r"<img [^>]*class=\"only-dark\"[^>]*>",
        "",
        content,
        flags=re.MULTILINE,
    )

    # Remove a tags that have class "only-dark"
    content = re.sub(
        r"<a [^>]*class=\"only-dark\"[^>]*>((?:(?!<\/a>).)|\s)*<\/a>\n",
        "",
        content,
        flags=re.MULTILINE,
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(content)


preprocess_readme("README.md", "PROCESSED_README.md")
