from pur import update_requirements
from urllib.request import urlopen
import re
from enum import Enum


class Device(Enum):
    CPU = "cpu"
    GPU = "gpu"


def _input_file_name_from_device(device: Device) -> str:
    if device == Device.GPU:
        return "requirements/optional_gpu.txt"
    else:
        return "requirements/optional.txt"


def _get_current_version_from_updates(updates: dict, package_name: str) -> str:
    return updates[package_name][0]["current"].public


def _get_latest_version_from_updates(updates: dict, package_name: str) -> str:
    return updates[package_name][0]["latest"].public


def _filter_urls_by_device(urls: list[str], device: Device) -> list[str]:
    filter: str = "cpu" if device == Device.CPU else "cu"
    return [url for url in urls if filter in url]


def _is_package_updated(updates: dict, package_name: str) -> bool:
    return updates[package_name][0]["updated"]


def _get_pyg_urls(updates: dict, device: Device) -> list[str]:
    def _scrape_pyg_org() -> str:
        url = "https://data.pyg.org/whl/"
        page = urlopen(url)
        html_bytes = page.read()
        return html_bytes.decode("utf-8")

    decoded_html = _scrape_pyg_org()
    latest_version = _get_latest_version_from_updates(updates, package_name="torch")

    def _extract_latest_torch_hyperlinks_from_html(
        decoded_html: str, torch_version: str
    ) -> list[str]:
        pattern = (
            f'<a href="torch-{torch_version}.*?.html">torch-{torch_version}.*?</a><br/>'
        )
        return re.findall(pattern, decoded_html, re.IGNORECASE)

    hyperlinks = _extract_latest_torch_hyperlinks_from_html(
        decoded_html, latest_version
    )

    def _get_urls_from_hyperlinks(hyperlinks: list[str]) -> list[str]:
        urls: list[str] = []
        for hyperlink in hyperlinks:
            match = re.search(r'href="([^"]+)"', hyperlink)
            if match:
                href = match.group(1)
                converted_string = f"https://data.pyg.org/whl/{href}"
                urls.append(converted_string)
        return urls

    urls = _get_urls_from_hyperlinks(hyperlinks)
    return _filter_urls_by_device(urls, device)


def _update_torch(updates: dict, device: Device):
    input_file_name: str = _input_file_name_from_device(device)

    with open(input_file_name, "r") as input_file:
        lines = input_file.readlines()

    def _remove_old_pyg_urls_from_lines(lines: list[str]) -> list[str]:
        target_pattern = r"^-f\s+https://data.pyg.org/whl/torch-.*\.html$"
        return list(filter(lambda line: not re.search(target_pattern, line), lines))

    lines = _remove_old_pyg_urls_from_lines(lines)

    def _remove_device_specific_versioning(lines: list[str]) -> list[str]:
        pattern = re.compile(r"(torch(?:-scatter)?==\d+\.\d+\.\d+)\S*(.*)")
        return [re.sub(pattern, r"\1\2", line) for line in lines]

    lines = _remove_device_specific_versioning(lines)

    def _get_torch_line_index_from_lines(lines: list[str]) -> int:
        return next(
            (i for i, line in enumerate(lines) if re.search("torch==", line)), -1
        )

    torch_line_index: int = _get_torch_line_index_from_lines(lines)
    new_urls = _get_pyg_urls(updates, device)

    def _prepare_urls_for_requirements_file(urls: list[str]) -> list[str]:
        return list(map(lambda url: "-f " + url + "\n", urls))

    new_urls = _prepare_urls_for_requirements_file(new_urls)

    # Insert new urls above the torch definition
    lines[torch_line_index:torch_line_index] = new_urls

    with open(input_file_name, "w") as file:
        file.writelines(lines)


def _update_requirements_for(device: Device):
    print(f"Updating requirements for ${device}")
    input_file_name: str = _input_file_name_from_device(device)
    updates: dict = update_requirements(input_file=input_file_name)

    if updates:
        if _is_package_updated(updates, "torch"):
            _update_torch(updates, device)


if __name__ == "__main__":
    for device in list(Device):
        _update_requirements_for(device)
