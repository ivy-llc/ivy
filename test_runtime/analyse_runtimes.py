# global
import os
import csv
import json
import numpy as np
import collections

FWS = ['numpy', 'tensorflow', 'torch', 'jax', 'mxnd']
DIM = '{}'.format(int(1e4))
os.makedirs('csvs/{}/'.format(DIM), exist_ok=True)
RATIO_MODE = 'mean'


def _reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def main():

    # Read Files #
    # -----------#

    this_dir = os.path.dirname(os.path.realpath(__file__))
    core_log_dir = os.path.join(this_dir, 'test_core_runtime/runtime_analysis/{}'.format(DIM))
    core_submodule_dirs = [os.path.join(core_log_dir, submodule_dir) for submodule_dir in os.listdir(core_log_dir)]
    nn_log_dir = os.path.join(this_dir, 'test_nn_runtime/runtime_analysis/{}'.format(DIM))
    nn_submodule_dirs = [os.path.join(nn_log_dir, submodule_dir) for submodule_dir in os.listdir(nn_log_dir)]
    submodule_dirs = core_submodule_dirs + nn_submodule_dirs
    filepaths = [os.path.join(submodule_dir, fname) for submodule_dir in submodule_dirs if submodule_dir[-5:] != '.json'
                 for fname in os.listdir(submodule_dir) if fname[0] != '_']
    filepaths.sort()
    all_results_dict = dict()
    for fpath in filepaths:
        method_name = fpath.split('/')[-1].split('.')[0]
        these_results_dict = dict()
        with open(fpath, 'r') as f:
            lines = [l for l in f.readlines() if l != '\n']
        f = None
        run = -1
        for l in lines:
            if 'tb0:' in l:
                run += 1
                these_results_dict[f][run] = dict()
            if l[0:13] == "<module 'ivy.":
                f = l[13:].split("'")[0]
                these_results_dict[f] = dict()
                run = -1
            elif 'end of analysis' in l:
                break
            else:
                entry_key = l.split(': ')[0]
                entry_val = l.split(': ')[1][:-1]
                if entry_key not in these_results_dict[f][run]:
                    these_results_dict[f][run][entry_key] = list()
                these_results_dict[f][run][entry_key].append(float(entry_val))

        all_results_dict[method_name] = these_results_dict

    # Read Reimplemented Functions #
    # -----------------------------#

    reimplemented_dict = {'jax': {}, 'mxnd': {}, 'numpy': {}, 'tensorflow': {}, 'torch': {}}
    json_filepaths = [item for item in submodule_dirs if item[-5:] == '.json']
    for json_filepath in json_filepaths:
        with open(json_filepath, 'r') as file:
            loaded_dict = json.loads(file.read())
        reimplemented_dict['jax'] = {**reimplemented_dict['jax'], **loaded_dict['jax']}
        reimplemented_dict['mxnd'] = {**reimplemented_dict['mxnd'], **loaded_dict['mxnd']}
        reimplemented_dict['numpy'] = {**reimplemented_dict['numpy'], **loaded_dict['numpy']}
        reimplemented_dict['tensorflow'] = {**reimplemented_dict['tensorflow'], **loaded_dict['tensorflow']}
        reimplemented_dict['torch'] = {**reimplemented_dict['torch'], **loaded_dict['torch']}

    # Extract Times #
    # --------------#

    for method, method_dict in all_results_dict.items():
        for f, f_dict in method_dict.items():

            # total times
            if 'tt0' in f_dict[0] and 'tt1' in f_dict[0]:
                total_times_ = np.asarray([run['tt1'][0] - run['tt0'][0] for run in f_dict.values()])
                total_times = _reject_outliers(total_times_)
                if len(total_times) == 0:
                    raise Exception('No data left after outlier rejection, consider increasing m.')
                total_mean_time = np.mean(total_times)
            else:
                raise Exception('Total times do not appear to be logged.')

            # backend times
            if 'tb1' in f_dict[0] and 'tb2' in f_dict[0]:
                backend_times_ = np.asarray([np.sum(np.asarray(run['tb2']) - np.asarray(run['tb1']))
                                             for run in f_dict.values()])
            elif 'tb3' in f_dict[0] and 'tb4' in f_dict[0]:
                backend_times_ = np.asarray([run['tb4'][0] - run['tb3'][0] for run in f_dict.values()])
            elif 'tb0' in f_dict[0] and 'tb4' in f_dict[0]:
                backend_times_ = total_times_
            else:
                raise Exception('Not enough backend times logged to compute runtimes.')
            backend_times = _reject_outliers(backend_times_)
            if len(backend_times) == 0:
                raise Exception('No data left after outlier rejection, consider increasing m.')
            backend_mean_time = np.mean(backend_times)

            # overhead times
            if 'to0' in f_dict[0] and 'to1' in f_dict[0]:
                overhead_times_ = np.asarray([np.sum(np.asarray(run['to1']) - np.asarray(run['to0']))
                                              for run in f_dict.values()])
                overhead_times = _reject_outliers(overhead_times_)
                if len(overhead_times) == 0:
                    raise Exception('No data left after outlier rejection, consider increasing m.')
                overhead_mean_time = np.mean(overhead_times)
            else:
                overhead_mean_time = 0

            # override if it's a re-implementation
            if method in reimplemented_dict[f] and reimplemented_dict[f][method] == [None]:
                backend_mean_time = total_mean_time
                overhead_mean_time = 0

            # total time
            total_mean_time = max(total_mean_time, backend_mean_time + overhead_mean_time)

            # results
            all_results_dict[method][f].clear()
            all_results_dict[method][f]['times'] = dict()
            all_results_dict[method][f]['times']['total_time'] = total_mean_time
            all_results_dict[method][f]['times']['backend_time'] = backend_mean_time
            all_results_dict[method][f]['times']['overhead_time'] = overhead_mean_time
            all_results_dict[method][f]['times']['graph_construct_time'] = max(total_mean_time - backend_mean_time - overhead_mean_time, 0)

            all_results_dict[method][f]['ratios'] = dict()
            backend_ratio = min(max(backend_mean_time / total_mean_time, 0), 1)
            all_results_dict[method][f]['ratios']['backend_ratio'] = backend_ratio
            ivy_overhead_ratio = min(max(overhead_mean_time / total_mean_time, 0), 1)
            all_results_dict[method][f]['ratios']['ivy_overhead_ratio'] = ivy_overhead_ratio
            graph_construct_ratio = min(max((total_mean_time - backend_mean_time - overhead_mean_time) / total_mean_time, 0), 1)
            all_results_dict[method][f]['ratios']['graph_construct_ratio'] = graph_construct_ratio

        # mean times across frameworks
        all_results_dict[method]['mean'] = dict()
        all_results_dict[method]['mean']['times'] = dict()
        all_results_dict[method]['mean']['times']['total_time'] =\
            np.mean(np.asarray([all_results_dict[method][fw]['times']['total_time'] for fw in FWS
                                if fw in all_results_dict[method]]))
        all_results_dict[method]['mean']['times']['backend_time'] =\
            np.mean(np.asarray([all_results_dict[method][fw]['times']['backend_time'] for fw in FWS
                                if fw in all_results_dict[method]]))
        all_results_dict[method]['mean']['times']['overhead_time'] =\
            np.mean(np.asarray([all_results_dict[method][fw]['times']['overhead_time'] for fw in FWS
                                if fw in all_results_dict[method]]))
        all_results_dict[method]['mean']['times']['graph_construct_time'] =\
            np.mean(np.asarray([all_results_dict[method][fw]['times']['graph_construct_time'] for fw in FWS
                                if fw in all_results_dict[method]]))

        # mean ratios across frameworks
        all_results_dict[method]['mean']['ratios'] = dict()
        all_results_dict[method]['mean']['ratios']['backend_ratio'] =\
            np.mean(np.asarray([all_results_dict[method][fw]['ratios']['backend_ratio'] for fw in FWS
                                if fw in all_results_dict[method]]))
        all_results_dict[method]['mean']['ratios']['ivy_overhead_ratio'] =\
            np.mean(np.asarray([all_results_dict[method][fw]['ratios']['ivy_overhead_ratio'] for fw in FWS
                                if fw in all_results_dict[method]]))
        all_results_dict[method]['mean']['ratios']['graph_construct_ratio'] =\
            np.mean(np.asarray([all_results_dict[method][fw]['ratios']['graph_construct_ratio'] for fw in FWS
                                if fw in all_results_dict[method]]))

    # Library Overhead #
    # -----------------#

    library_usage_fpath = 'library_usage.json'
    if os.path.exists(library_usage_fpath):
        lib_overhead_dict = dict()
        with open(library_usage_fpath, 'r') as file:
            usage_dict = json.loads(file.read())
        for lib_name, lib_dict in usage_dict.items():
            total_runtime = 0
            backend_runtime = 0
            overhead_runtime = 0
            graph_construct_runtime = 0
            for method_name, method_occurance in lib_dict.items():
                if method_name == 'Container':
                    continue
                tot = all_results_dict[method_name]['mean']['times']['total_time']
                total_runtime += method_occurance * tot if not np.isnan(tot) else 0
                back = all_results_dict[method_name]['mean']['times']['backend_time']
                backend_runtime += method_occurance * back if not np.isnan(back) else 0
                over = all_results_dict[method_name]['mean']['times']['overhead_time']
                overhead_runtime += method_occurance * over if not np.isnan(over) else 0
                graph = all_results_dict[method_name]['mean']['times']['graph_construct_time']
                graph_construct_runtime += method_occurance * graph if not np.isnan(graph) else 0
            compiled_ratio = overhead_runtime/total_runtime
            eager_ratio = (overhead_runtime + graph_construct_runtime)/total_runtime
            print('\n{}:\neager_percentage: {}\ncompiled_percentage: {}\n'.format(lib_name, eager_ratio*100, compiled_ratio*100))

    # Save Results #
    # -------------#
    for key_to_save in FWS + ['mean']:
        all_results_dict_ordered = collections.OrderedDict(
            sorted(all_results_dict.items(),
                   key=lambda key_n_val: -key_n_val[1][key_to_save]['ratios']['ivy_overhead_ratio']
                                         - key_n_val[1][key_to_save]['ratios']['graph_construct_ratio']
                   if key_to_save in key_n_val[1] else 0.))
        with open('csvs/{}/'.format(DIM) + key_to_save + '_runtime_analysis.csv', 'w+') as file:
            csv_writer = csv.writer(file)
            for method, res_dict in all_results_dict_ordered.items():
                if key_to_save not in res_dict:
                    continue
                csv_writer.writerow([method, str(res_dict[key_to_save]['ratios']['ivy_overhead_ratio']*100),
                                     str(res_dict[key_to_save]['ratios']['graph_construct_ratio'] * 100),
                                     str(res_dict[key_to_save]['ratios']['backend_ratio'] * 100),
                                     '',
                                     str(res_dict[key_to_save]['times']['overhead_time']*1000),
                                     str(res_dict[key_to_save]['times']['graph_construct_time'] * 1000),
                                     str(res_dict[key_to_save]['times']['backend_time']*1000)])


if __name__ == '__main__':
    main()
