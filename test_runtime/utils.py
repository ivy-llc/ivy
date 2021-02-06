"""
Utility functions for performing runtime analysis
"""

# global
import os
import time
import collections

TIMES_DICT = collections.OrderedDict()


def append_to_file(fname, string):
    with open(fname, 'a+') as f:
        f.write('\n' + string + '\n')


def log_time(fpath, time_name, time_in=None, time_at_start=False):
    count = time.perf_counter() if time_at_start else None
    if fpath not in TIMES_DICT:
        TIMES_DICT[fpath] = collections.OrderedDict()
    if time_name not in TIMES_DICT[fpath]:
        TIMES_DICT[fpath][time_name] = list()
    if time_in is not None:
        TIMES_DICT[fpath][time_name].append(time_in)
    elif time_at_start:
        TIMES_DICT[fpath][time_name].append(count)
    else:
        TIMES_DICT[fpath][time_name].append(time.perf_counter())


def write_times():
    for fpath, fpath_dict in TIMES_DICT.items():
        time_names = fpath_dict.keys()
        time_lists = fpath_dict.values()
        for time_vals in zip(*time_lists):
            for time_name, time_val in zip(time_names, time_vals):
                with open(fpath, 'a') as f:
                    f.write('\n{}: {}\n'.format(time_name, time_val))
    TIMES_DICT.clear()


def modify_ivy_file(filepath, this_dir, offsets, namespace_dict, dim):

    # import log_time and get filename
    module_name = filepath.split('/')[-1][:-3]
    with open(filepath, 'r', errors='ignore') as f:
        original_src_lines = f.readlines()
    fw = filepath.split('/')[-3]
    def_key = 'def '
    all_methods = [line.split(def_key)[-1].split('(')[0]
                   for i, line in enumerate(original_src_lines) if def_key in line]
    all_def_indices = [i for i, line in enumerate(original_src_lines) if def_key in line]
    ivy_methods = [method for method in all_methods if method[0] != '_']
    new_src_lines = original_src_lines
    new_src_lines.insert(3, 'from test_runtime.utils import log_time\n')
    log_dir = os.path.join(this_dir, 'runtime_analysis/{}/{}'.format(dim, module_name))
    for i, (method, def_idx) in enumerate(zip(all_methods, all_def_indices)):
        new_src_lines.insert(def_idx+i+2, '    fname = "{}"\n'.format(os.path.join(log_dir, '{}.txt'.format(method))))

    # add timing points for backend
    ivy_def_indices = [i for i, line in enumerate(new_src_lines) if def_key in line and def_key + '_' not in line]
    return_key = 'return '
    found_a_return = False
    num_added_to_file = 0
    for i, (method, def_idx) in enumerate(zip(ivy_methods, ivy_def_indices)):
        ret_idxs = list()
        idx = def_idx + num_added_to_file
        while True:
            if idx == len(new_src_lines):
                break
            new_line = new_src_lines[idx]
            if 'return ' in new_line or 'return\n' in new_line:
                found_a_return = True
                spaces_str = new_line.split('return')[0]
                new_src_lines.insert(idx, spaces_str + 'log_time(fname, "tb3")\n')
                num_added_to_file += 1
                idx += 1
                ret_idxs.append(idx)
            elif 'raise ' in new_line:
                found_a_return = True
            elif not new_line.strip():
                if not found_a_return:
                    raise Exception('return should be found before white space.')
                if method in offsets[fw]:
                    for j, offset in enumerate(offsets[fw][method]):
                        if offset is None:
                            offset = 0
                            skip_val = 1
                        else:
                            skip_val = 2
                        new_idx = ret_idxs[-1] + offset - j*2
                        new_idx -= len([ret_idx for ret_idx in ret_idxs if ret_idx > new_idx])
                        spaces_str = ' ' * (len(new_src_lines[new_idx]) - len(new_src_lines[new_idx].lstrip(' ')))
                        new_src_lines.insert(new_idx, spaces_str + 'log_time(fname, "tb1")\n')
                        new_src_lines.insert(new_idx + skip_val, spaces_str + 'log_time(fname, "tb2", time_at_start=True)\n')
                        num_added_to_file += 2
                        for k, ret_idx in enumerate(ret_idxs):
                            if ret_idx >= new_idx + skip_val:
                                ret_idxs[k] += 1
                            if ret_idx >= new_idx:
                                ret_idxs[k] += 1
                break
            idx += 1

    # add timing points for overhead
    num_added_to_file = 0
    group_start_idx = None
    grouping = False
    final_src_lines = new_src_lines.copy()
    prev_line = new_src_lines[0]
    for i, line in enumerate(new_src_lines):
        idx = i + num_added_to_file
        overhead_line = ((max([item in line for item in namespace_dict[fw]])
                          or '.astype(' in line or '.type(' in line or '.reshape(' in line or '.permute(' in line)
                         and 'elif' not in line and 'with' not in line and 'isinstance' not in line
                         and '__dict__' not in line and ' return ' not in line and '    if ' not in line
                         and 'log_time(fname, "tb1")' not in prev_line and line[0] == ' ')
        if overhead_line and not grouping:
            grouping = True
            group_start_idx = idx
        elif not overhead_line and grouping:
            group_end_idx = idx - 1
            grouping = False
            spaces_str = ' ' * (len(final_src_lines[group_start_idx]) - len(final_src_lines[group_start_idx].lstrip(' ')))
            final_src_lines.insert(group_start_idx, spaces_str + 'log_time(fname, "to0")\n')
            if '    if ' in final_src_lines[group_end_idx + 1]:
                spaces_str = ' ' * (len(final_src_lines[group_end_idx + 2]) - len(final_src_lines[group_end_idx + 2].lstrip(' ')))
            else:
                spaces_str = ' ' * (len(final_src_lines[group_end_idx + 1]) - len(final_src_lines[group_end_idx + 1].lstrip(' ')))
            final_src_lines.insert(group_end_idx + 2, spaces_str + 'log_time(fname, "to1", time_at_start=True)\n')
            num_added_to_file += 2
        prev_line = line

    # save file
    with open(filepath, 'w') as f:
        f.writelines(final_src_lines)
