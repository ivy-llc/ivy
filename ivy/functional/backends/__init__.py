# Gets dtype from a version dictionary
def dtype_from_version(dic, version):
    # if version is a dict, extract the version
    if isinstance(version, dict):
        version = version["version"]

    # If key is already in the dictionary, return the value
    if version in dic:
        return dic[version]

    version_tuple = tuple(map(int, version.split('.')))

    # If key is not in the dictionary, check if it's in any range
    # three formats are supported:
    # 1. x.y.z and above
    # 2. x.y.z and below
    # 3. x.y.z to x.y.z
    for key in dic.keys():
        kl = key.split(" ")
        k1 = tuple(map(int, kl[0].split('.')))

        if "above" in key and k1 <= version_tuple:
            return dic[key]
        if "below" in key and k1 >= version_tuple:
            return dic[key]
        if "to" in key and k1 <= version_tuple <= tuple(map(int, kl[2].split('.'))):
            return dic[key]

    raise ValueError(f"No dtype found for version {version}")