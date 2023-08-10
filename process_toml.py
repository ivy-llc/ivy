import re
import toml


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

def parse_requirements(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        requirements = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if '# mod_name=' in line:
                    name = line.split('# mod_name=')[1].strip()
                    requirement = line.split(' #')[0].strip()
                else:
                    requirement = line.split(' #')[0].strip()
                    name = requirement.split('=')[0] if '==' in requirement else requirement.split('<')[0] if '<=' in requirement else requirement
                    name = name.split('>')[0] if '>=' in requirement else name
                requirements[name] = requirement
        return requirements



# Read the version from _version.py
with open("ivy/_version.py") as f:
    version_code = f.read().strip()
    version = version_code.split("=")[1].strip().strip('"')

# Read the existing pyproject.toml file
pyproject_data = toml.load("pyproject.toml")

# Update the version
pyproject_data["project"]["version"] = version

# Ensure that the project.dependencies key exists
if "dependencies" not in pyproject_data["project"]:
    pyproject_data["project"]["dependencies"] = {}

# Parse the requirements from requirements.txt and update the dependencies
requirements = parse_requirements("requirements/requirements.txt")
for name, requirement in requirements.items():
    pyproject_data["project"]["dependencies"][name] = requirement

# Write the updated pyproject.toml file back to disk
with open("pyproject.toml", "w") as f:
    toml.dump(pyproject_data, f)

# Preprocess the README
preprocess_readme("README.md", "PROCESSED_README.md")