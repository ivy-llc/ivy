def dtype_from_version(dic,version):
    for key in dic.keys():
        if tuple(key.strip().split(" ")[0].split('.'))==tuple(version.split('.')):
            return dic[key]
        if "above" in key and tuple(key.strip().split(" ")[0].split('.'))<=tuple(version.split('.')):
            return dic[key]
        if "below" in key and tuple(key.strip().split(" ")[0].split('.'))>=tuple(version.split('.')):
            return dic[key]
        if "to" in key and (tuple(key.strip().split(" ")[0].split('.'))>=tuple(version.split('.'))) and tuple(key.strip().split(" ")[2].split('.'))<=tuple(version.split('.')):
            return dic[key]
