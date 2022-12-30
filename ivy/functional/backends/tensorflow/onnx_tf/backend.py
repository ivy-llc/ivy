"""Backend for running ONNX on Tensorflow

To run this, you will need to have Tensorflow installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
  from itertools import izip as zip
except ImportError:  # will be 3.x series
  pass

from onnx import defs
from onnx import numpy_helper
from onnx.backend.base import Backend
from onnx.backend.base import namedtupledict
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
from onnx.helper import make_opsetid
import tensorflow as tf
import numpy as np

from onnx_tf.backend_rep import TensorflowRep
from onnx_tf.common import data_type
from onnx_tf.common import get_unique_suffix
from onnx_tf.common import supports_device as common_supports_device
from onnx_tf.common.handler_helper import get_all_backend_handlers
from onnx_tf.pb_wrapper import OnnxNode
from onnx_tf.backend_tf_module import BackendTFModule, TFModule
import onnx_tf.common as common

training_flag_name = "_onnx_tf_internal_is_training"


class TensorflowBackend(Backend):
  """ Tensorflow Backend for ONNX
  """

  @classmethod
  def prepare(cls,
              model,
              device='CPU',
              strict=True,
              logging_level='INFO',
              auto_cast=False,
              **kwargs):
    """Prepare an ONNX model for Tensorflow Backend.

    This function converts an ONNX model to an internel representation
    of the computational graph called TensorflowRep and returns
    the converted representation.

    param model: The ONNX model to be converted.
    param device: The device to execute this model on. It can be either CPU (default) or CUDA.
    param strict: Whether to enforce semantic equivalence between the original model
      and the converted tensorflow model, defaults to True (yes, enforce semantic equivalence).
      Changing to False is strongly discouraged.
      Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.
    param logging_level: The logging level, default is INFO. Change it to DEBUG
      to see more conversion details or to WARNING to see less
    param auto_cast: Whether to auto cast data types that might lose precision for the tensors
      with types not natively supported by Tensorflow, default is False

    returns: A TensorflowRep class object representing the ONNX model
    """
    super(TensorflowBackend, cls).prepare(model, device, **kwargs)
    common.logger.setLevel(logging_level)
    common.logger.handlers[0].setLevel(logging_level)
    common.sys_config.auto_cast = auto_cast
    common.sys_config.device = device

    return cls.onnx_model_to_tensorflow_rep(model, strict, **kwargs)

  @classmethod
  def onnx_model_to_tensorflow_rep(cls, model, strict, **kwargs):
    """ Convert ONNX model to TensorflowRep.

    param model: ONNX ModelProto object.
    param strict: whether to enforce semantic equivalence between the original model
      and the converted tensorflow model.
    return: TensorflowRep object.
    """

    # Models with IR_VERSION less than 3 does not have opset_import set.
    # We default to minimum opset, this behavior is consistent with
    # onnx checker.
    # c.f. https://github.com/onnx/onnx/blob/427ac0c1b792363d373e3d7e4eef97fa46458420/onnx/checker.cc#L478
    if model.ir_version < 3:
      opset_import = [make_opsetid(defs.ONNX_DOMAIN, 1)]
    else:
      opset_import = model.opset_import
    return cls._onnx_graph_to_tensorflow_rep(model.graph, opset_import, strict,
                                             **kwargs)

  @classmethod
  def _onnx_graph_to_tensorflow_rep(cls, graph_def, opset, strict, **kwargs):
    """ Convert ONNX graph to TensorflowRep.

    param graph_def: ONNX GraphProto object.
    param opset: ONNX OperatorSetIdProto list.
    param strict: whether to enforce semantic equivalence between the original model
      and the converted tensorflow model.
    kwargs: additional arguements to generate tensor_dict for model debugging
    return: TensorflowRep object.
    """
    # To generate tensor_dict or not, default is False
    gen_tensor_dict = kwargs[
        'gen_tensor_dict'] if 'gen_tensor_dict' in kwargs else False
    # User provided input tensors, in the case the model inputs have unknown shapes
    input_tensor_dict = kwargs[
        'input_tensor_dict'] if 'input_tensor_dict' in kwargs else dict()
    training_mode = kwargs[
        'training_mode'] if 'training_mode' in kwargs else False

    handlers = cls._get_handlers(opset)

    # initializer: TensorProtos representing the values to initialize
    # a given tensor.
    # initialized: A list of names of the initialized tensors.

    if graph_def.initializer:
      initialized = {init.name for init in graph_def.initializer}
    else:
      initialized = set()

    input_dict = dict()

    module = BackendTFModule(handlers, opset, strict, graph_def, cls)
    signatures = dict()
    if training_mode:
      tf_rep_graph = kwargs['graph'] if 'graph' in kwargs else tf.Graph()
    else:
      tf_rep_graph = tf.Graph()
    with tf_rep_graph.as_default():
      for value_info in graph_def.input:
        if value_info.name in initialized or not value_info.type.HasField(
            'tensor_type'):
          continue
        shape = list(
            d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None
            for d in value_info.type.tensor_type.shape.dim)
        value_info_name = value_info.name.replace(
            ":", "_tf_") + "_" + get_unique_suffix(
            ) if ":" in value_info.name else value_info.name

        tf_spec = tf.TensorSpec(
            shape, data_type.onnx2tf(value_info.type.tensor_type.elem_type),
            value_info_name)
        signatures[value_info.name] = tf_spec

        if gen_tensor_dict or training_mode:
          x = tf.compat.v1.placeholder(
              data_type.onnx2tf(value_info.type.tensor_type.elem_type),
              name=value_info_name,
              shape=shape
          ) if value_info.name not in input_tensor_dict else input_tensor_dict[
              value_info.name]
          input_dict[value_info.name] = x

      if gen_tensor_dict or training_mode:
        input_dict_items = cls._onnx_initializer_to_input_dict_items(
            graph_def.initializer, training_mode=True)
        tensor_dict = dict(input_dict)
        tensor_dict.update(input_dict_items)
        tensor_dict[training_flag_name] = tf.compat.v1.placeholder_with_default(
            False, shape=[])
        for node in graph_def.node:
          onnx_node = OnnxNode(node)
          output_ops = cls._onnx_node_to_tensorflow_op(onnx_node,
                                                       tensor_dict,
                                                       handlers,
                                                       opset=opset,
                                                       strict=strict)
          curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
          tensor_dict.update(curr_node_output_map)

    tf_rep = TensorflowRep()
    tf_rep.inputs = [
        value_info.name
        for value_info in graph_def.input
        if value_info.name not in initialized
    ]
    tf_rep.outputs = [value_info.name for value_info in graph_def.output]
    module.outputs = tf_rep.outputs
    tf_rep.tf_module = module
    tf_rep.signatures = signatures
    if gen_tensor_dict or training_mode:
      tf_rep.tensor_dict = tensor_dict
    if training_mode:
      tf_rep.graph = tf_rep_graph
    tf_rep.onnx_op_list = cls._get_onnx_op_list(graph_def)
    return tf_rep

  @classmethod
  def _get_onnx_op_list(cls, graph_def):
    """ Get ONNX operator counts of the model.

    param graph_def: ONNX GraphProto object.
    return: Dictionary of all operators counts in the model.
    """

    def get_onnx_op_from_graph_and_subgraph(graph, op_list):
      for node in graph.node:
        op_list[node.op_type] = 1 if node.op_type not in op_list.keys(
        ) else op_list[node.op_type] + 1
        if node.op_type in ['Loop', 'Scan']:
          onnx_node = OnnxNode(node)
          body = onnx_node.attrs["body"]
          op_list = get_onnx_op_from_graph_and_subgraph(body, op_list)
        elif node.op_type == 'If':
          onnx_node = OnnxNode(node)
          then_branch = onnx_node.attrs['then_branch']
          op_list = get_onnx_op_from_graph_and_subgraph(then_branch, op_list)
          else_branch = onnx_node.attrs['else_branch']
          op_list = get_onnx_op_from_graph_and_subgraph(else_branch, op_list)
      return op_list

    op_list = get_onnx_op_from_graph_and_subgraph(graph_def, dict())
    sorted_op_list = dict()
    for key in sorted(op_list):
      sorted_op_list[key] = op_list[key]
    return sorted_op_list

  @classmethod
  def run_node(cls, node, inputs, device='CPU', outputs_info=None, **kwargs):
    """ Run ONNX node.

    param node: ONNX NodeProto object.
    param inputs: Inputs.
    param device: Device run on.
    param outputs_info: None.
    param kwargs: Other args.
    return: Outputs.
    """

    super(TensorflowBackend, cls).run_node(node, inputs, device)
    common.sys_config.device = device

    node = OnnxNode(node)
    input_tensors = []
    for i in inputs:
      if i is None:
        input_tensors.append(i)
      else:
        input_tensors.append(tf.constant(i))

    if isinstance(inputs, dict):
      feed_dict_raw = inputs
    else:
      assert len(node.inputs) == len(inputs)
      feed_dict_raw = dict(zip(node.inputs, inputs))

    # TODO: is constant the best way for feeding inputs?
    input_dict = {}
    for k, v in feed_dict_raw.items():
      if isinstance(v, list):
        list_input = []
        for x in v:
          if x is None:
            list_input.append(x)
          else:
            list_input.append(tf.constant(x))
        input_dict[k] = list_input
      elif v is None:  # keep None for empty optional data
        input_dict[k] = v
      else:
        input_dict[k] = tf.constant(v)

    module = TFModule(node, cls)

    output_vals = module(**input_dict)
    output_vals = [
        val.numpy() if isinstance(val, tf.Tensor) else val
        for val in output_vals
    ]

    return namedtupledict('Outputs', node.outputs)(*output_vals)

  @classmethod
  def _onnx_initializer_to_input_dict_items(cls,
                                            initializer,
                                            training_mode=False):
    """ Convert ONNX graph initializer to input dict items.

    param initializer: ONNX graph initializer, list of TensorProto.
    return: List of input dict items.
    """

    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      return numpy_helper.to_array(onnx_tensor).flatten().tolist()

    def validate_initializer_name(name):
      # Prepend a unique suffix if leading character is "_"
      name = get_unique_suffix() + name if name[0] == "_" else name

      # Replace ":" with "_tf_" and append a unique suffix for
      # traceability
      return name.replace(
          ":", "_tf_") + "_" + get_unique_suffix() if ":" in name else name

    if training_mode:
      tensor_dict = [
          (init.name,
           tf.Variable(np.array(tensor2list(init)).reshape(init.dims),
                       shape=init.dims,
                       dtype=data_type.onnx2tf(init.data_type),
                       name=validate_initializer_name(init.name)))
          for init in initializer
      ]
    else:
      tensor_dict = [(init.name,
                      tf.constant(tensor2list(init),
                                  shape=init.dims,
                                  dtype=data_type.onnx2tf(init.data_type),
                                  name=validate_initializer_name(init.name)))
                     for init in initializer]

    return tensor_dict

  @classmethod
  def _onnx_node_to_tensorflow_op(cls,
                                  node,
                                  tensor_dict,
                                  handlers=None,
                                  opset=None,
                                  strict=True):
    """
    Convert onnx node to tensorflow op.

    Args:
      node: Onnx node object.
      tensor_dict: Tensor dict of graph.
      opset: Opset version of the operator set. Default 0 means using latest version.
      strict: whether to enforce semantic equivalence between the original model
        and the converted tensorflow model, defaults to True (yes, enforce semantic equivalence).
        Changing to False is strongly discouraged.
    Returns:
      Tensorflow op
    """
    handlers = handlers or cls._get_handlers(opset)
    if handlers:
      handler = handlers[node.domain].get(
          node.op_type, None) if node.domain in handlers else None
      if handler:
        return handler.handle(node, tensor_dict=tensor_dict, strict=strict)

    raise BackendIsNotSupposedToImplementIt("{} is not implemented.".format(
        node.op_type))

  @classmethod
  def _get_handlers(cls, opset):
    """ Get all backend handlers with opset.

    param opset: ONNX OperatorSetIdProto list.
    return: All backend handlers.
    """
    opset = opset or [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
    opset_dict = dict([(o.domain, o.version) for o in opset])
    return get_all_backend_handlers(opset_dict)

  @classmethod
  def supports_device(cls, device):
    return common_supports_device(device)

  @classmethod
  def onnx_graph_to_tensorflow_ops(cls,
                                   subgraph,
                                   tensor_dict,
                                   opset=None,
                                   strict=True):
    """
    Converts ONNX graph to Tensorflow operations
    Args:
      subgraph:         the ONNX graph to be converted.
      tensor_dict:      tensor dict of the subgraph.
      opset:            opset version of the operator set.
      strict:           whether to enforce semantic equivalence between the
                        original model and the converted tensorflow model,
                        defaults to True (yes, enforce semantic equivalence).
    Returns:
      array of Tensorflow Tensors
    """
    for node in subgraph.node:
      onnx_node = OnnxNode(node)
      output_ops = cls._onnx_node_to_tensorflow_op(onnx_node,
                                                   tensor_dict,
                                                   opset=opset,
                                                   strict=strict)
      curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
      tensor_dict.update(curr_node_output_map)
    return tensor_dict

  @classmethod
  def onnx_graph_to_tensorflow_rep(cls, graph_def, strict=True, **kwargs):
    """
    Converts ONNX graph to TensorflowRep
    Args:
      graph_def:        the ONNX graph to be converted
      strict:           whether to enforce semantic equivalence between the
                        original model and the converted tensorflow model,
                        defaults to True (yes, enforce semantic equivalence).
    Returns:
      TensorflowRep object.
    """
    # get the opset of the installed ONNX
    opset = [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
    return cls._onnx_graph_to_tensorflow_rep(graph_def, opset, strict, **kwargs)


prepare = TensorflowBackend.prepare

run_node = TensorflowBackend.run_node

run_model = TensorflowBackend.run_model

supports_device = TensorflowBackend.supports_device

onnx_graph_to_tensorflow_ops = TensorflowBackend.onnx_graph_to_tensorflow_ops

onnx_graph_to_tensorflow_rep = TensorflowBackend.onnx_graph_to_tensorflow_rep
