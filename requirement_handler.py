# flake8: noqa
from typing import List
import operator
import requests
import sys

url = "https://pypi.python.org/pypi/"


def tilde(x, y):
    if len(x) != len(y):
        return None
    for i in range(len(x) - 1):
        if x[i] != y[i]:
            return None
    return True


def lower_of_two(x, y):
    return x if x < y else y


def higher_of_two(x, y):
    return x if x > y else y


def make_len_equal(x, y):
    if len(x) < len(y):
        x = list(x)
        while len(x) != len(y):
            x.append("0")
        return (tuple(x), y)
    else:
        y = list(y)
        while len(y) != len(x):
            y.append("0")
        return (x, tuple(y))


def inti(a):
    try:
        return int(a)
    except:
        return a


def higher_or_lower(x, y, operator, operator1):
    acceptable_version = None
    if len(x) != len(y):
        x, y = make_len_equal(x, y)
    x = tuple(map(inti, x))
    y = tuple(map(inti, y))
    if string_dict[operator](x, y):
        if operator in lower:
            acceptable_version = lower_of_two(x, y)
        else:
            acceptable_version = higher_of_two(x, y)

    if string_dict[operator1](y, x):
        if operator1 in lower:
            if acceptable_version:
                acceptable_version = higher_of_two(
                    acceptable_version, lower_of_two(y, x)
                )
            else:
                acceptable_version = lower_of_two(y, x)
        else:
            if acceptable_version:
                acceptable_version = higher_of_two(
                    acceptable_version, higher_of_two(y, x)
                )
            else:
                acceptable_version = higher_of_two(y, x)
    return acceptable_version


string_dict = {
    "~=": tilde,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

lower = ["<=", "<"]


def get_version_from_string(s):
    s = s.replace("(", "").replace(")", "").split(",")
    # s is now a list with version orderings for eg: [<2.0,>=1.37.0]
    lis = []
    for i in s:
        string = ""
        for j in range(len(i)):
            while not str.isdigit(i[j]):
                string = string + i[j]
                j += 1
            if string not in string_dict:
                raise ValueError(f"string_dict doesn't have {string}")
            lis.append({string: tuple(i[j:].split("."))})
            break
    return lis


def get_package_versions(lis: List):
    if not lis:
        return dict()
    dic = dict()
    for i in lis:
        if ";" in i:
            i = i[: i.index(";")]
        if " " in i:
            dic[i.split(" ")[0]] = get_version_from_string(i.split(" ")[1])
        else:
            dic[i] = ()

    return dic


def resolution(conflict_keys, dic1, dic2):
    resolution = {}
    for i in conflict_keys:
        if dic1[i] == dic2[i]:
            continue
        for j in dic1[i]:
            for k in dic2[i]:
                resolution[i] = higher_or_lower(
                    k[list(k.keys())[0]],
                    j[list(j.keys())[0]],
                    list(j.keys())[0],
                    list(k.keys())[0],
                )

    return resolution


def package_conflicts(pkg1, pkg2=None):
    data_1 = get_package_requirements(pkg1)
    data_2 = get_package_requirements(pkg2)
    if not data_1 or not data_2:
        return None
    data_1_dic = (
        [] if not data_1["info"]["requires_dist"] else data_1["info"]["requires_dist"]
    )
    data_2_dic = (
        [] if not data_2["info"]["requires_dist"] else data_2["info"]["requires_dist"]
    )
    data_1_dic = get_package_versions(
        ["python " + str(data_1["info"]["requires_python"])] + data_1_dic
    )
    data_2_dic = get_package_versions(
        ["python " + str(data_2["info"]["requires_python"])] + data_2_dic
    )
    return resolution(data_1_dic.keys() & data_2_dic.keys(), data_1_dic, data_2_dic)


def get_package_requirements(pkg):
    return requests.get(url + str(pkg) + "/json").json()


def tuple_to_version(tup):
    s = ""
    for i in tup:
        s += str(i) + "."
    return s[:-1]


# example usage
# if __name__ == "__main__":
#     fw1=sys.argv[1]
#     fw2=sys.argv[2]
#     python=sys.argv[3]
#     conf=package_conflicts(fw1,fw2)
#     if not conf:
#         sys.exit("-1")
#     if 'python' in conf:
#         if tuple_to_version(conf['python'])>python:
#             conf['python']=[conf['python'],-1]
#         elif tuple_to_version(conf['python'])<python:
#             conf['python']=[conf['python'],-2]
#
#         else: del conf['python']
#     f=open("additional_requirements.txt","w")
#     for key,val in conf.items():
#         if key=='python':
#             if val[-1]==-1:
#                 print(f"The environment has python {python} which is lower than the required version {tuple_to_version(val[0])} do anticipate unexpected results")
#             else:
#                 print(
#                     f"The environment has python {python} which is greater than the required version {tuple_to_version(val[0])} do anticipate unexpected results")
#
#         f.write(f"{key}=={val}")


print(package_conflicts("h5py/3.7.0", "h5py/3.7.0"))
# #
# # print(get_package_requirements('tensorflow/2.2.0')["info"]["requires_dist"])
# print(get_package_requirements('torch/1.5.0')["info"]["requires_python"])
