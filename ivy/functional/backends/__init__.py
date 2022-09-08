def dtype_from_version(dic,version):
    for i in dic.keys():
        if i.strip(' ')[0]==str(version):
            return dic[i]
        if "above" in i and int(str(version).strip('.'))>=int(i.strip(' ')[0].strip('.')):
            return dic[i]
        if "below" in i and int(str(version).strip('.'))<=int(i.strip(' ')[0].strip('.')):
            return dic[i]
        if "to" in i and (int(str(version).strip('.'))>=int(i.strip(' ')[0].strip('.')) and int(str(version).strip('.'))<=int(i.strip(' ')[2].strip('.'))):
            return dic[i]

