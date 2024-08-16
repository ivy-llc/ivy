import os

if __name__ == "__main__":
    tag = os.environ["TAG"]
    python_tag, abi_tag, plat_name = tag.split("-")
    if os.path.exists("dist"):
        for file in os.listdir("dist"):
            old_name = f"{python_tag}-none-{plat_name}.whl"
            new_name = f"{python_tag}-{abi_tag}-{plat_name}.whl"
            if file.endswith(old_name):
                os.rename(
                    os.path.join("dist", file),
                    os.path.join("dist", file[: -len(old_name)] + new_name),
                )
