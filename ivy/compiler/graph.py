# global
import os
import ivy
import cv2
import copy
import random
import inspect
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# local
from ivy.compiler.param import Param
from ivy.compiler import globals as glob
# noinspection PyProtectedMember
from ivy.compiler.helpers import _get_shape, _get_id, _terminal_pids_to_key, _args_str_from_fn, _output_str_from_fn,\
    _param_to_label, _copy_func


class Graph:

    # noinspection PyProtectedMember
    def __init__(self, fn, *args, stateful=None, **kwargs):

        # stateful
        self._stateful = ivy.default(stateful, [])
        self._stateful_classes = tuple([x.__class__ for x in self._stateful])
        self._stateful_param_ids = [id(x) for x in self._stateful]
        self._stateful_param_shapes = [_get_shape(x) for x in self._stateful]

        # function being compiled into a graph
        self._fn = fn

        # function args and kwargs
        self._fn_signature = dict(inspect.signature(self._fn).parameters)

        # positional args
        self._args = list(args)
        self._arg_tracked_idxs = ivy.nested_indices_where(
            args, lambda a: ivy.is_array(a) or isinstance(a, self._stateful_classes))
        self._arg_param_ids = [_get_id(a) for a in ivy.multi_index_nest(args, self._arg_tracked_idxs)]
        self._arg_param_types = [a.__class__ for a in ivy.multi_index_nest(args, self._arg_tracked_idxs)]
        self._arg_param_shapes = [_get_shape(a) for a in ivy.multi_index_nest(args, self._arg_tracked_idxs)]

        # key-word args
        self._kwargs = kwargs
        self._kwarg_tracked_idxs = ivy.nested_indices_where(
            kwargs, lambda v: ivy.is_array(v) or isinstance(v, self._stateful_classes))
        self._kwarg_param_ids = [_get_id(v) for v in ivy.multi_index_nest(kwargs, self._kwarg_tracked_idxs)]
        self._kwarg_param_types = [v.__class__ for v in ivy.multi_index_nest(kwargs, self._kwarg_tracked_idxs)]
        self._kwarg_param_shapes = [_get_shape(a) for a in ivy.multi_index_nest(kwargs, self._kwarg_tracked_idxs)]

        # output param ids
        self._output = None  # initialized during op logging
        self._output_tracked_idxs = None  # initialized during op logging
        self._output_param_ids = list()

        # op logging storage
        self._pid_to_functions_dict = dict()

        # temporary sub-graph storage
        self._tmp_sub_param_dict = dict()
        self._tmp_sub_functions = list()
        self._num_subgrahs = 0

        # graph storage
        self._param_dict = dict()
        self._functions = dict()
        self._num_functions = dict()
        self._max_subgraph_widths = dict()
        self._max_subgraph_heights = dict()

        # connected flag
        self._outer_connected = False
        self._all_connected = False

        # grouped functions
        self._grouped_functions = dict()

        # all functions
        self._all_functions_fixed = list()

        # graph formatting
        self._inter_node_color = (0., 0.8, 0.)
        self._stateful_node_color = (0.9, 0.7, 0.2)
        self._io_node_color = (0.4, 0.4, 1.)
        self._edge_color = (0., 0.4, 0.)
        self._node_size = 300
        self._linewidths = 1.
        self._arrow_size = 10
        self._cv2_font_size = 12
        self._plt_font_size = 12
        self._input_functions = dict()
        self._output_functions = dict()
        self._labels_layer = None
        self._labels_mask = None
        self._x_limits = (0, 1)
        self._y_limits = (0, 1)
        # ToDo: remove the need for saving temporary images tmp_img.png
        #  if fig.canvas.tostring_rgb or similar can return correct formatting
        self._img_tmp_fpath = os.path.join(ivy.tmp_dir(), 'tmp_img.png')
        self._border_divisor = 10
        self._border_divisor_p2 = self._border_divisor + 2
        self._canvas_scale = self._border_divisor / self._border_divisor_p2
        self._canvas_normed_origin = np.array([1 / self._border_divisor_p2, 1 / self._border_divisor_p2])
        self._default_dpi = 2000
        self._max_dpi = 4000
        self._dpi = self._default_dpi
        matplotlib.rcParams['figure.dpi'] = self._dpi

    # Properties #
    # -----------#

    @property
    def _all_grouped_functions(self):
        # ToDo: make this order more optimal, in the same manner by which the each sub-graph order is optimal
        all_grouped_functions = list()
        for gfs in self._grouped_functions.values():
            for i, fs in enumerate(gfs):
                if len(all_grouped_functions) == i:
                    all_grouped_functions.append(list())
                all_grouped_functions[i] += fs
        return all_grouped_functions

    @property
    def _all_param_dict(self):
        all_param_dict = dict()
        for pd in self._param_dict.values():
            all_param_dict = {**all_param_dict, **pd}
        return all_param_dict

    @property
    def _all_functions(self):
        return [i for sl in self._all_grouped_functions for i in sl]

    @property
    def _max_graph_width(self):
        return max([len(fns) for fns in self._all_grouped_functions]) if self._all_grouped_functions else 0

    @property
    def _max_graph_height(self):
        return len(self._all_grouped_functions)

    # Foward with Op Logging #
    # -----------------------#

    def _compute_return(self):
        ret = self._fn(*self._args, **self._kwargs)
        if not isinstance(ret, tuple):
            ret = (ret,)
        return ret

    def _register_output(self, ret):
        self._output = list(ret)
        self._output_tracked_idxs = ivy.nested_indices_where(
            ret, lambda x: ivy.is_array(x) or isinstance(x, self._stateful_classes))
        output_tracked_idxs = ivy.nested_indices_where(
            ret, lambda x: ivy.is_array(x) or isinstance(x, self._stateful_classes))
        # noinspection PyProtectedMember
        self._output_param_ids = [_get_id(x) for x in ivy.multi_index_nest(list(ret), output_tracked_idxs)]

        # find any inputs which were fed directly to the output, and update pid and add identity function
        for i, pid in enumerate(self._output_param_ids):
            if pid in self._arg_param_ids + self._kwarg_param_ids:

                new_pid = random.randint(0, 2 ** 48)

                def new_fn(a, _):
                    return a[0]

                new_fn.arg_param_ids = [pid]
                new_fn.arg_tracked_idxs = [[0]]
                new_fn.kwarg_tracked_idxs = list()
                new_fn.kwarg_param_ids = list()
                new_fn.kwarg_param_types = list()
                new_fn.kwarg_param_shapes = list()
                new_fn.output_param_ids = [new_pid]
                if pid in self._arg_param_ids:
                    index = self._arg_param_ids.index(pid)
                    arg_param_types = [self._arg_param_types[index]]
                    arg_param_shapes = [self._arg_param_shapes[index]]
                    output_param_type = self._arg_param_types[index]
                    output_param_shape = self._arg_param_shapes[index]
                else:
                    index = self._kwarg_param_ids.index(pid)
                    arg_param_types = [self._kwarg_param_types[index]]
                    arg_param_shapes = [self._kwarg_param_shapes[index]]
                    output_param_type = self._kwarg_param_types[index]
                    output_param_shape = self._kwarg_param_shapes[index]
                new_fn.arg_param_types = arg_param_types
                new_fn.arg_param_shapes = arg_param_shapes
                new_fn.output_param_types = [output_param_type]
                new_fn.fns_in = list()
                new_fn.output_tracked_idxs = [[0]]
                new_fn.output_param_shapes = [output_param_shape]
                new_fn.terminal = True
                new_fn.is_constant = False

                self.add_fn_to_dict(new_pid, new_fn)
                self._output_param_ids[i] = new_pid

    # noinspection PyProtectedMember
    def log_all_ops(self):
        glob.op_logging = True
        self._register_output(self._compute_return())
        glob.op_logging = False

    # Getters and Setters #
    # --------------------#

    # inference

    def get_param(self, pid):
        return self._tmp_sub_param_dict[pid].get()

    def set_param(self, pid, value):
        self._tmp_sub_param_dict[pid].set(value)

    # compiling

    def add_param(self, pid, ptype, tree_height, shape=None):
        self._tmp_sub_param_dict[pid] = Param(ptype, tree_height, shape)

    def increment_param_count(self, pid):
        self._tmp_sub_param_dict[pid].set_count(self._tmp_sub_param_dict[pid].count + 1)

    def add_fn_to_dict(self, pid, fn):
        self._pid_to_functions_dict[pid] = fn

    def get_param_recursive(self, pid, depth, receiving_fn=None):
        if pid in self._tmp_sub_param_dict:
            return
        if pid in self._pid_to_functions_dict:
            fn = self._pid_to_functions_dict[pid]
        else:
            if not ivy.exists(receiving_fn):
                idx = self._output_param_ids.index(pid)
                del self._output_tracked_idxs[idx]
                del self._output_param_ids[idx]
                return
            if pid in receiving_fn.arg_param_ids:
                idx = receiving_fn.arg_param_ids.index(pid)
                del receiving_fn.arg_tracked_idxs[idx]
                del receiving_fn.arg_param_ids[idx]
                del receiving_fn.arg_param_types[idx]
                del receiving_fn.arg_param_shapes[idx]
            if pid in receiving_fn.kwarg_param_ids:
                idx = receiving_fn.kwarg_param_ids.index(pid)
                del receiving_fn.kwarg_tracked_idxs[idx]
                del receiving_fn.kwarg_param_ids[idx]
                del receiving_fn.kwarg_param_types[idx]
                del receiving_fn.kwarg_param_shapes[idx]
            receiving_fn.is_constant = len(receiving_fn.arg_param_ids + receiving_fn.kwarg_param_ids) == 0
            return
        [self.get_param_recursive(pid, depth + 1, fn) for pid in copy.copy(fn.arg_param_ids)]
        [self.get_param_recursive(pid, depth + 1, fn) for pid in copy.copy(fn.kwarg_param_ids)]
        [self.increment_param_count(pid) for pid in fn.arg_param_ids + fn.kwarg_param_ids]
        fn.tree_depth = depth
        if fn.is_constant:
            for recv_fn in fn.fns_out:
                for pid in fn.output_param_ids:
                    if pid in recv_fn.arg_param_ids:
                        idx = recv_fn.arg_param_ids.index(pid)
                        del recv_fn.arg_tracked_idxs[idx]
                        del recv_fn.arg_param_ids[idx]
                        del recv_fn.arg_param_types[idx]
                        del recv_fn.arg_param_shapes[idx]
                    if pid in recv_fn.kwarg_param_ids:
                        idx = recv_fn.kwarg_param_ids.index(pid)
                        del recv_fn.kwarg_tracked_idxs[idx]
                        del recv_fn.kwarg_param_ids[idx]
                        del recv_fn.kwarg_param_types[idx]
                        del recv_fn.kwarg_param_shapes[idx]
                recv_fn.is_constant = len(recv_fn.arg_param_ids + recv_fn.kwarg_param_ids) == 0
            fn.output_tracked_idxs.clear()
            fn.output_param_ids.clear()
            fn.output_param_types.clear()
            fn.output_param_shapes.clear()
        else:
            self._tmp_sub_functions.append(fn)
            [self.add_param(pid, ptype, depth, shape)
             for pid, ptype, shape in zip(fn.output_param_ids, fn.output_param_types, fn.output_param_shapes)]
        return

    # debugging

    def params_all_empty(self):
        return min([len(param) == 0 for param in self._tmp_sub_param_dict.values()]) is True

    # Function creation #
    # ------------------#

    def _chain_functions(self, terminal_pids, assert_non_empty=True):

        # dict key
        dict_key = _terminal_pids_to_key(terminal_pids)

        # add input params to param dict
        [self.add_param(pid, ptype, 'leaf', shape) for pid, ptype, shape in
         zip(self._arg_param_ids, self._arg_param_types, self._arg_param_shapes)]
        [self.add_param(pid, ptype, 'leaf', shape) for pid, ptype, shape in
         zip(self._kwarg_param_ids, self._kwarg_param_types, self._kwarg_param_shapes)]

        # add stateful params to param dict
        [self.add_param(pid, ptype, 'leaf', shape) for pid, ptype, shape in
         zip(self._stateful_param_ids, self._stateful_classes, self._stateful_param_shapes)]

        # recursively chain the graph via backward traversal
        [self.get_param_recursive(pid, depth=0) for pid in terminal_pids]
        [self.increment_param_count(pid) for pid in terminal_pids]

        # assert there are some functions in the graph
        if assert_non_empty:
            assert self._tmp_sub_functions, 'Tried to chain functions for an empty graph'
        elif not self._tmp_sub_functions:
            return

        # sort the param ids based on depth, in order of input to output
        max_depth = max([p.depth if isinstance(p.depth, int) else 1 for p in self._tmp_sub_param_dict.values()])
        for k, v in self._tmp_sub_param_dict.items():
            if v.depth == 'leaf':
                # noinspection PyProtectedMember
                self._tmp_sub_param_dict[k]._tree_depth = max_depth + 1

        self._tmp_sub_param_dict = {k: v for k, v in sorted(self._tmp_sub_param_dict.items(), key=lambda knv: -knv[1].depth)}

        # save this sub-graph in the param dict
        self._param_dict[dict_key] = self._tmp_sub_param_dict

        # function for storing function heights
        def store_fn_heights(fn):
            if hasattr(fn, 'tree_height'):
                return fn.tree_height
            heights_in = [store_fn_heights(fn_in) for fn_in in fn.fns_in if fn_in in self._tmp_sub_functions]
            if heights_in:
                _height = max(heights_in) + 1
            else:
                _height = 0
            fn.tree_height = _height
            return _height

        # store function heights
        [store_fn_heights(self._pid_to_functions_dict[pid]) for pid in terminal_pids]

        # find the height of the tree
        max_tree_height = max([fn.tree_height for fn in self._tmp_sub_functions]) if self._tmp_sub_functions else -1

        # group the functions based on their height in the tree from the starting leaf nodes
        grouped_functions = list()
        for height in range(0, max_tree_height+1):
            fns = [fn for fn in self._tmp_sub_functions if fn.tree_height == height]
            if height == 0:
                fns = sorted(fns, key=lambda x: len(x.fns_in))
            else:
                fns_hm1 = grouped_functions[-1]
                # noinspection PyUnresolvedReferences
                leftmost_idxs =\
                    [max(enumerate([fn in fn_hm1.fns_out for fn_hm1 in fns_hm1 if hasattr(fn_hm1, 'fns_out')]),
                         key=lambda x: x[1])[0] for fn in fns]
                fns = [fn for fn, _ in sorted(zip(fns, leftmost_idxs), key=lambda x: x[1])]
            grouped_functions.append(fns)
        self._grouped_functions[dict_key] = grouped_functions

        # stack functions in the best order
        self._tmp_sub_functions = [i for sl in grouped_functions for i in sl]

        # update the total graph storage
        self._num_functions[dict_key] = len(self._tmp_sub_functions)
        self._functions[dict_key] = self._tmp_sub_functions

        # compute maximum width and height of the graph
        self._max_subgraph_widths[dict_key] =\
            max([len(fns) for fns in self._grouped_functions]) if self._grouped_functions else 0
        self._max_subgraph_heights[dict_key] = len(self._grouped_functions)

    def _call(self, *args, **kwargs):
        # ToDo: make this as efficient as possible; this is performed at runtime
        [self.set_param(pid, ivy.index_nest(args, idx))
         for pid, idx in zip(self._arg_param_ids, self._arg_tracked_idxs)]
        [self.set_param(pid, ivy.index_nest(kwargs, idx))
         for pid, idx in zip(self._kwarg_param_ids, self._kwarg_tracked_idxs)]
        # ToDo: change so continual resetting of fixed stateful objects as below is not required
        [self.set_param(pid, val)
         for pid, val in zip(self._stateful_param_ids, self._stateful)]
        for i, fn in enumerate(self._all_functions_fixed):
            arg_vals = [self.get_param(pid) for pid in fn.arg_param_ids]
            kwarg_vals = [self.get_param(pid) for pid in fn.kwarg_param_ids]
            ret = fn(arg_vals, kwarg_vals)
            if not isinstance(ret, tuple):
                ret = (ret,)
            [self.set_param(pid, ivy.index_nest(ret, idx))
             for pid, idx in zip(fn.output_param_ids, fn.output_tracked_idxs)]
        ret_vals = [self.get_param(pid) for pid in self._output_param_ids]
        ivy.set_nest_at_indices(self._output, self._output_tracked_idxs, ret_vals)
        if len(self._output) == 1:
            return self._output[0]
        return self._output

    def connect(self, output_connected_only=True):
        self._chain_functions(self._output_param_ids)
        self._num_subgrahs = 1
        if not output_connected_only:
            for pid, fn in self._pid_to_functions_dict.items():
                if fn.terminal:
                    self._chain_functions(fn.output_param_ids)
                    self._num_subgrahs += 1
            self._all_connected = True
        self._outer_connected = True

    def compiled(self):
        if not self._outer_connected:
            self.connect()
        self._all_functions_fixed = self._all_functions
        return self._call

    # Graph Visualization #
    # --------------------#

    def _is_stateful(self, f):
        if hasattr(f, 'args'):
            for a in ivy.multi_index_nest(f.args, f.arg_tracked_idxs):
                if isinstance(a, self._stateful_classes):
                    return True
        if hasattr(f, 'kwargs'):
            for kwa in ivy.multi_index_nest(f.kwargs, f.kwarg_tracked_idxs):
                if isinstance(kwa, self._stateful_classes):
                    return True
        return False

    def _position_nodes(self, g, num_inputs, num_outputs, all_nodes, randomness_factor):

        pos_dict = dict()
        assert 0 <= randomness_factor <= 1

        # select position based on width and height of graph
        for height, nodes in enumerate(all_nodes):
            width = len(nodes)
            for w, n in enumerate(nodes):
                pos = np.array([(height+1) / (self._max_graph_height+1), 0.5 if width == 1 else w / (width - 1)])
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                h_delta = 0.5/self._max_graph_height
                h_rand = np.random.uniform(-h_delta, h_delta)
                w_delta = 0.5 if width == 1 else 0.5/(width-1)
                w_delta_low = 0 if (w == 0 and width != 1) else -w_delta
                w_delta_high = 0 if (w == (width-1) and width != 1) else w_delta
                w_rand = np.random.uniform(w_delta_low, w_delta_high)
                pos += np.array([h_rand, w_rand]) * randomness_factor
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                pos_dict[n] = pos

        # add inputs
        if num_inputs > 0:
            input_idx = 0
            for n in g.nodes:
                if n not in pos_dict and n[1].__name__[0:7] == 'input: ':
                    pos = np.array([0., 0.5 if num_inputs == 1 else input_idx/(num_inputs-1)])
                    assert np.logical_and((0 <= pos), (pos <= 1)).all()
                    h_delta = 0.5/self._max_graph_height
                    h_rand = np.random.uniform(0, h_delta)
                    w_delta = 0.5 if num_inputs == 1 else 0.5/(num_inputs-1)
                    w_delta_low = 0 if input_idx == 0 else -w_delta
                    w_delta_high = 0 if input_idx == (num_inputs-1) else w_delta
                    w_rand = np.random.uniform(w_delta_low, w_delta_high)
                    pos += np.array([h_rand, w_rand]) * randomness_factor
                    assert np.logical_and((0 <= pos), (pos <= 1)).all()
                    pos_dict[n] = pos
                    input_idx += 1

        # add outputs
        if num_outputs > 0:
            output_idx = 0
            for n in g.nodes:
                if n not in pos_dict and n[1].__name__ == 'output':
                    pos = np.array([1., 0.5 if num_outputs == 1 else output_idx/(num_outputs-1)])
                    assert np.logical_and((0 <= pos), (pos <= 1)).all()
                    h_delta = 0.5/self._max_graph_height
                    h_rand = np.random.uniform(-h_delta, 0)
                    w_delta = 0.5 if num_outputs == 1 else 0.5/(num_outputs-1)
                    w_delta_low = 0 if output_idx == 0 else -w_delta
                    w_delta_high = 0 if output_idx == (num_outputs-1) else w_delta
                    w_rand = np.random.uniform(w_delta_low, w_delta_high)
                    pos += np.array([h_rand, w_rand]) * randomness_factor
                    assert np.logical_and((0 <= pos), (pos <= 1)).all()
                    pos_dict[n] = pos
                    output_idx += 1

        return pos_dict

    def _add_edge(self, g, func, pid_in, idx, inp, num_inputs):
        if pid_in in self._pid_to_functions_dict:
            fn_in = self._pid_to_functions_dict[pid_in]
            fn_pid = fn_in.output_param_ids[0]
        elif pid_in in self._input_functions:
            fn_in = self._input_functions[pid_in]
            fn_pid = pid_in
            num_inputs += 1
        else:
            fn_in = _copy_func(inp)
            idx0 = idx[0]
            sig = list(self._fn_signature.keys())
            if isinstance(idx0, str):
                arg_name = idx0
            elif 'args' not in sig and isinstance(idx0, int) and idx0 < len(sig):
                arg_name = sig[idx0]
            else:
                arg_name = str(idx0)
            fnc_name = 'input: ' + arg_name
            idx1on = idx[1:]
            if idx1on:
                fnc_name += ', {}'.format(idx1on)
            fn_in.__name__ = fnc_name
            fn_pid = pid_in
            self._input_functions[pid_in] = fn_in
            num_inputs += 1
        start_args = _args_str_from_fn(fn_in)
        start_output = _output_str_from_fn(fn_in)
        start_node = (fn_pid, fn_in, start_args, start_output)
        end_args = _args_str_from_fn(func)
        end_output = _output_str_from_fn(func)
        end_node = (func.output_param_ids[0], func, end_args, end_output)
        g.add_edge(start_node, end_node)
        return num_inputs

    def _draw_labels_plt(self, g, pos, with_arg_labels, with_output_labels):

        # draw node labels
        nx.draw_networkx_labels(
            g, pos=pos, labels={n: n[1].__name__.replace('_', '') for n in g.nodes}, font_size=self._plt_font_size)

        # maybe add function arg labels
        if with_arg_labels:
            font_size = self._plt_font_size / 2
            nx.draw_networkx_labels(
                g, pos={k: v + np.array([0., font_size/100]) for k, v in pos.items()}, font_size=font_size,
                font_color=(0., 100/255, 0.), labels={n: n[2] for n in g.nodes})

        # maybe add function output labels
        if with_output_labels:
            font_size = self._plt_font_size / 2
            nx.draw_networkx_labels(
                g, pos={k: v - np.array([0., font_size/100]) for k, v in pos.items()}, font_size=font_size,
                font_color=(0., 100/255, 0.), labels={n: n[3] for n in g.nodes})

    def _draw_labels_cv2(self, g, pos, img_dims, with_arg_labels, with_output_labels):

        # get rgb image
        img = np.full(img_dims + [3], 255, dtype=np.uint8)

        # canvas dims
        canvas_dims = np.asarray(img_dims)
        canvas_dims_pix = np.flip(canvas_dims, -1)

        # draw node labels
        for node, label in {node: node[1].__name__.replace('_', '') for node in g.nodes}.items():
            normed_pos = pos[node]
            canvas_normed_pos = (normed_pos * self._canvas_scale) + self._canvas_normed_origin
            thickness = max(int(round(self._cv2_font_size)), 1)
            textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self._cv2_font_size, thickness)[0]
            txt_offset = np.array([-textsize[0]/2, textsize[1]/2])
            pixel_pos = canvas_normed_pos * canvas_dims_pix
            pixel_pos = np.array([pixel_pos[0], canvas_dims_pix[1] - pixel_pos[1]])
            pixel_pos = np.clip(np.round(pixel_pos + txt_offset), 0, canvas_dims_pix).astype(np.int32)
            cv2.putText(img, label, pixel_pos, cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=self._cv2_font_size,
                        thickness=thickness)

        # maybe add function arg labels
        if with_arg_labels:
            font_size = self._cv2_font_size / 2
            increment = (font_size/120) * (self._default_dpi/self._dpi)
            for node, normed_pos in {node: v + np.array([0., increment]) for node, v in pos.items()}.items():
                label = node[2]
                canvas_normed_pos = (normed_pos * self._canvas_scale) + self._canvas_normed_origin
                thickness = max(int(round(font_size)), 1)
                lines = label.split('\n')
                num_lines = len(lines)
                for i, line in enumerate(lines):
                    textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
                    txt_offset = np.array([-textsize[0] / 2, (i - num_lines + 1) * (textsize[1]) + textsize[1] / 2])
                    pixel_pos = canvas_normed_pos * canvas_dims_pix
                    pixel_pos = np.array([pixel_pos[0], canvas_dims_pix[1] - pixel_pos[1]])
                    pixel_pos = np.clip(np.round(pixel_pos + txt_offset), 0, canvas_dims_pix).astype(np.int32)
                    cv2.putText(img, line, pixel_pos, cv2.FONT_HERSHEY_SIMPLEX, color=(0, 100, 0), fontScale=font_size,
                                thickness=thickness)

        # maybe add function output labels
        if with_output_labels:
            font_size = self._cv2_font_size / 2
            increment = (font_size / 120) * (self._default_dpi / self._dpi)
            for node, normed_pos in {k: v - np.array([0., increment]) for k, v in pos.items()}.items():
                label = node[3]
                canvas_normed_pos = (normed_pos * self._canvas_scale) + self._canvas_normed_origin
                thickness = max(int(round(font_size)), 1)
                lines = label.split('\n')
                for i, line in enumerate(lines):
                    textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
                    txt_offset = np.array([-textsize[0]/2, i*textsize[1] + textsize[1]/2])
                    pixel_pos = canvas_normed_pos * canvas_dims_pix
                    pixel_pos = np.array([pixel_pos[0], canvas_dims_pix[1] - pixel_pos[1]])
                    pixel_pos = np.clip(np.round(pixel_pos + txt_offset), 0, canvas_dims_pix).astype(np.int32)
                    cv2.putText(img, line, pixel_pos, cv2.FONT_HERSHEY_SIMPLEX, color=(0, 100, 0), fontScale=font_size,
                                thickness=thickness)

        # mask
        mask = np.logical_not(np.min(img == np.array([255, 255, 255]), -1))

        # return
        return img, mask

    def _show_for_functions(self, ax, functions, with_edge_labels, with_arg_labels, with_output_labels,
                            output_connected_only, randomness_factor, format_graph, cv2_labels=True, pos=None):

        # create directed networkX graph
        g = nx.DiGraph()

        # add input and intermediate nodes
        def inp():
            pass

        num_inputs = 0

        for func in self._pid_to_functions_dict.values():
            if func not in functions and output_connected_only:
                continue
            for pid_in, idx in zip(func.arg_param_ids, func.arg_tracked_idxs):
                num_inputs = self._add_edge(g, func, pid_in, idx, inp, num_inputs)
            for pid_in, idx in zip(func.kwarg_param_ids, func.kwarg_tracked_idxs):
                num_inputs = self._add_edge(g, func, pid_in, idx, inp, num_inputs)

        # assert all positions are accounted for, if provided
        if ivy.exists(pos):
            assert min([n in pos for n in g.nodes])

        # add output nodes
        def out():
            pass

        out.__name__ = 'output'

        num_outputs = 0

        for pid_in in self._output_param_ids:
            if pid_in not in self._output_functions:
                self._output_functions[pid_in] = _copy_func(out)
            fn_in = self._pid_to_functions_dict[pid_in]
            fn_pid = fn_in.output_param_ids[0]
            start_args = _args_str_from_fn(fn_in)
            start_output = _output_str_from_fn(fn_in)
            start_node = (fn_pid, fn_in, start_args, start_output)
            fn_out = self._output_functions[pid_in]
            end_args = _args_str_from_fn(fn_out)
            end_output = _output_str_from_fn(fn_out)
            end_node = (fn_pid, fn_out, end_args, end_output)
            g.add_edge(start_node, end_node)
            num_outputs += 1

        # position nodes
        all_nodes = list()
        max_graph_width = 0
        for fns in self._all_grouped_functions:
            nodes = set([(f.output_param_ids[0], f, _args_str_from_fn(f), _output_str_from_fn(f)) for f in fns])
            max_graph_width = max(max_graph_width, len(nodes))
            all_nodes.append(nodes)
        max_dim = max(max_graph_width, self._max_graph_height + 2)
        pos = ivy.default(pos, self._position_nodes(g, num_inputs, num_outputs, all_nodes, randomness_factor))
        pos = {n: p for n, p in pos.items() if n in g.nodes}

        # formatting

        if format_graph:
            self._dpi = np.clip(max_dim * 50, 300, self._max_dpi)
            ax.figure.dpi = self._dpi
            self._node_size = 500 / max_dim
            self._linewidths = 1 / max_dim
            self._arrow_size = max(10 / max_dim, 1)
            self._plt_font_size = 36 / max_dim
            self._cv2_font_size = min((50 / max_dim) * (self._dpi/self._default_dpi), self._dpi/200)

        # draw nodes

        # input
        input_nodes = [n for n in g.nodes if n[1].__name__[0:7] == 'input: ']
        max_graph_width = max(max_graph_width, len(input_nodes))
        input_pos = {n: pos[n] for n in g.nodes if n[1].__name__[0:7] == 'input: '}

        nx.draw_networkx_nodes(g, input_pos, input_nodes, node_color=[self._io_node_color]*len(input_nodes),
                               node_shape='s', node_size=[self._node_size]*len(input_nodes),
                               linewidths=self._linewidths)

        # intermediate
        intermediate_nodes =\
            [n for n in g.nodes if (n[1].__name__[0:6] not in ['input:', 'output'] and not self._is_stateful(n[1]))]
        intermediate_pos =\
            {n: pos[n] for n in g.nodes if
             (n[1].__name__[0:6] not in ['input:', 'output'] and not self._is_stateful(n[1]))}

        nx.draw_networkx_nodes(g, intermediate_pos, intermediate_nodes,
                               node_color=[self._inter_node_color] * len(intermediate_nodes),
                               node_shape='s', node_size=[self._node_size]*len(intermediate_nodes),
                               linewidths=self._linewidths)

        # stateful
        stateful_nodes = [n for n in g.nodes if self._is_stateful(n[1])]
        stateful_pos = {n: pos[n] for n in g.nodes if self._is_stateful(n[1])}

        nx.draw_networkx_nodes(g, stateful_pos, stateful_nodes,
                               node_color=[self._stateful_node_color]*len(stateful_nodes), node_shape='s',
                               node_size=[self._node_size]*len(stateful_nodes), linewidths=self._linewidths)

        # output
        output_nodes = [n for n in g.nodes if n[1].__name__ == 'output']
        output_pos = {n: pos[n] for n in g.nodes if n[1].__name__ == 'output'}

        nx.draw_networkx_nodes(g, output_pos, output_nodes, node_color=[self._io_node_color]*len(output_nodes),
                               node_shape='s', node_size=[self._node_size]*len(output_nodes),
                               linewidths=self._linewidths)

        # draw edges
        nx.draw_networkx_edges(g, arrows=True, pos=pos, edge_color=[self._edge_color]*len(g.edges),
                               width=self._linewidths, arrowsize=self._arrow_size,
                               node_size=[self._node_size]*len(g.nodes), node_shape='s')

        # maybe add edge labels
        if with_edge_labels:
            edge_labels = dict()
            for edge in g.edges:
                node_in = edge[0]
                node_out = edge[1]
                node_in_pid = node_in[0]
                node_in_name = node_in[1].__name__
                node_out_pid = node_out[0]
                node_out_name = node_out[1].__name__
                if node_in_pid in self._pid_to_functions_dict:
                    base_fn = self._pid_to_functions_dict[node_in_pid]
                    base_pids = base_fn.output_param_ids
                    if node_out_name == 'output':
                        tip_pids = self._output_param_ids
                    else:
                        tip_fn = self._pid_to_functions_dict[node_out_pid]
                        tip_pids = tip_fn.arg_param_ids + tip_fn.kwarg_param_ids
                elif node_in_name[0:7] == 'input: ':
                    base_pids = self._arg_param_ids + self._kwarg_param_ids + self._stateful_param_ids
                    tip_fn = self._pid_to_functions_dict[node_out_pid]
                    tip_pids = tip_fn.arg_param_ids + tip_fn.kwarg_param_ids
                else:
                    raise Exception('node {} not found in self._pid_to_functions_dict,'
                                    'and is not of type input or output'.format(node_in))
                pids = [pid for pid in base_pids if pid in tip_pids]
                params = [self._tmp_sub_param_dict[pid] for pid in pids]
                edge_labels[edge] = '_'.join([_param_to_label(p) for p in params])
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=self._plt_font_size / 2)

        # format graph
        if format_graph:
            ax.set_aspect(max_graph_width / self._max_graph_height)
            x_border = 1 / self._border_divisor
            self._x_limits = (-x_border, 1 + x_border)
            y_border = 1 / self._border_divisor
            self._y_limits = (-y_border, 1 + y_border)
            ax.set_xlim(*self._x_limits)
            ax.set_ylim(*self._y_limits)
            plt.subplots_adjust(left=self._x_limits[0], right=self._x_limits[1], bottom=self._y_limits[0],
                                top=self._y_limits[1], hspace=0., wspace=0.)
            ax.margins(x=0, y=0, tight=True)

        # draw
        plt.draw_if_interactive()

        # labels
        if cv2_labels:

            # determine canvas size
            plt.savefig(self._img_tmp_fpath, bbox_inches='tight', pad_inches=0)
            img = cv2.imread(self._img_tmp_fpath, -1)
            img_dims = list(img.shape[0:2])

            # draw labels
            layer, mask = self._draw_labels_cv2(g, pos, img_dims, with_arg_labels, with_output_labels)
            if ivy.exists(self._labels_mask) and ivy.exists(self._labels_layer):
                self._labels_mask = np.logical_or(mask, self._labels_mask)
                self._labels_layer = np.where(mask, layer, self._labels_layer)
            else:
                self._labels_mask = mask
                self._labels_layer = layer
        else:
            self._draw_labels_plt(g, pos, with_arg_labels, with_output_labels)

        # return positions, in case subsequent graph highlighting is needed
        return pos

    def show(self, save_to_disk=False, with_edge_labels=True, with_arg_labels=True, with_output_labels=True,
             output_connected_only=True, randomness_factor=0.1, highlight_subgraph=None, cv2_labels=True, fname=None):

        # ensure graph is connected
        if not self._outer_connected or (not output_connected_only and not self._all_connected):
            self.connect()

        # matplotlib
        plt.cla()
        ax = plt.gca()
        ax.axis('off')
        fig = ax.figure
        fig.set_canvas(FigureCanvas(fig))

        # show for functions
        pos = self._show_for_functions(
            ax, self._all_functions, with_edge_labels, with_arg_labels, with_output_labels, output_connected_only,
            randomness_factor, True, cv2_labels)

        # maybe highlight sub-graph
        if isinstance(highlight_subgraph, int):

            # set node color as red
            self._inter_node_color = (0.8, 0., 0.)
            self._stateful_node_color = (0.8, 0., 0.)
            self._io_node_color = (0.8, 0., 0.)
            self._edge_color = (0.4, 0., 0.)

            # show highlighted sub-graph
            subgraph_pid = list(self._functions.keys())[highlight_subgraph]
            self._show_for_functions(
                ax, self._functions[subgraph_pid], with_edge_labels, with_arg_labels, with_output_labels, True,
                randomness_factor, False, cv2_labels, pos=pos)

        # reset node color as green
        self._inter_node_color = (0., 0.8, 0.)
        self._stateful_node_color = (0.9, 0.7, 0.2)
        self._io_node_color = (0.4, 0.4, 1.)
        self._edge_color = (0., 0.4, 0.)

        # show
        plt.subplots_adjust(left=self._x_limits[0], right=self._x_limits[1], bottom=self._y_limits[0],
                            top=self._y_limits[1], hspace=0., wspace=0.)
        ax.margins(x=0, y=0, tight=True)
        plt.show()

        # maybe save to disk
        if save_to_disk:
            fname = ivy.default(fname, 'graph_{}.png'.format(''.join(
                [f.__name__.replace('_', '')[0] for f in self._tmp_sub_functions][0:20])))
            if fname[-4:] != '.png':
                if '.' in fname:
                    fname = '.'.join(fname.split('.')[:-1])
                fname += '.png'
            if cv2_labels:
                plt.savefig(self._img_tmp_fpath, bbox_inches='tight', pad_inches=0)
                img = cv2.imread(self._img_tmp_fpath, -1)[..., 0:3]
                if ivy.exists(self._labels_mask) and ivy.exists(self._labels_layer):
                    img = np.where(np.expand_dims(self._labels_mask, -1), self._labels_layer, img)
                cv2.imwrite(fname, img)
            else:
                plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    # Clearing #
    # ---------#

    def clear(self):
        self._tmp_sub_param_dict.clear()
        self._tmp_sub_functions.clear()
