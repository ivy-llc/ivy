"""
Collection of Ivy neural network layers as stateful classes.
"""

# global
import ivy
import abc
from ivy.core.container import Container


# Base #
# -----#

class Layer(abc.ABC):

    def __init__(self, v=None):
        """
        Initialze Ivy layer, which is a stateful object consisting of trainable variables.

        :param v: Ivy container of trainable variables. Created internally by default.
        :type v: ivy container, optional.
        """
        if v is None:
            self.v = Container(self._create_variables())
        else:
            self.v = Container(v)

    # Public #

    def __call__(self, *args, v=None, **kwargs):
        """
        Run layer forward pass, by first setting the variables via either input or internal values, and then calling the
        overridden forward method.

        :param v: Ivy container of trainable variables. Internal variables used by default.
        :type v: ivy container, optional.
        """
        if v is None:
            v = self.v
        else:
            v = Container(v)
        return self._forward(*args, **kwargs, v=v)

    # Abstract #

    @abc.abstractmethod
    def _create_variables(self):
        """
        create internal trainable variables, and return as arbitrary nested dict.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        """
        the forward pass of the layer, called after handling the optional input variables.
        """
        raise NotImplementedError


# Linear #
# -------#

class Linear(Layer):

    def __init__(self, input_channels, output_channels, v=None):
        """
        Linear layer, also referred to as dense or fully connected. The layer receives tensors with input_channels last
        dimension and returns a new tensor with output_channels last dimension, following matrix multiplication with the
        weight matrix and addition with the bias vector.

        :param input_channels: Number of input channels for the layer.
        :type input_channels: int
        :param output_channels: Number of output channels for the layer.
        :type output_channels: int
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        Layer.__init__(self, v)

    def _create_variables(self):
        """
        Create internal variables for the Linear layer
        """
        # ToDo: support other initialization mechanisms, via class constructor options
        # ToDo: automate the consstruction of these variables, with helper functions
        wlim = (6 / (self._output_channels + self._input_channels)) ** 0.5
        w = ivy.variable(ivy.random_uniform(-wlim, wlim, (self._output_channels, self._input_channels)))
        b = ivy.variable(ivy.zeros([self._output_channels]))
        return {'w': w, 'b': b}

    def _forward(self, inputs, v):
        """
        Perform forward pass of the linear layer.

        :param inputs: Inputs to process *[batch_shape, in]*.
        :type inputs: array
        :param v: the variables for each of the lstm cells, as a container, use internal variables by default.
        :type v: ivy container of parameter arrays, optional
        :return: The outputs following the linear operation and bias addition *[batch_shape, out]*
        """
        return ivy.linear(inputs, v.w, v.b)


# LSTM #
# -----#

class LSTM(Layer):

    def __init__(self, input_channels, output_channels, num_layers=1, return_sequence=True, return_state=True, v=None):
        """
        LSTM layer, which is a set of stacked lstm cells.

        :param input_channels: Number of input channels for the layer
        :type input_channels: int
        :param output_channels: Number of output channels for the layer
        :type output_channels: int
        :param num_layers: Number of lstm cells in the lstm layer, default is 1.
        :type num_layers: int, optional
        :param return_sequence: Whether or not to return the entire output sequence, or just the latest timestep.
                                Default is True.
        :type return_sequence: bool, optional
        :param return_state: Whether or not to return the latest hidden and cell states. Default is True.
        :type return_state: bool, optional
        :param v: the variables for each of the lstm cells, as a container, constructed internally by default.
        :type v: ivy container of parameter arrays, optional
        """
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._num_layers = num_layers
        self._return_sequence = return_sequence
        self._return_state = return_state
        Layer.__init__(self, v)

    # Public #

    def get_initial_state(self, batch_shape):
        """
        Get the initial state of the hidden and cell states, if not provided explicitly
        """
        batch_shape = list(batch_shape)
        return ([ivy.zeros((batch_shape + [self._output_channels])) for i in range(self._num_layers)],
                [ivy.zeros((batch_shape + [self._output_channels])) for i in range(self._num_layers)])

    # Overridden

    def _create_variables(self):
        """
        Create internal variables for the LSTM layer
        """
        # ToDo: support other initialization mechanisms, via class constructor options
        # ToDo: automate the consstruction of these variables, with helper functions
        wlim = (6 / (self._output_channels + self._input_channels)) ** 0.5
        input_weights = dict(zip(
            ['layer_' + str(i) for i in range(self._num_layers)],
            [{'w': ivy.variable(ivy.random_uniform(
                -wlim, wlim, (self._input_channels if i == 0 else self._output_channels, 4 * self._output_channels)))}
                for i in range(self._num_layers)]))
        wlim = (6 / (self._output_channels + self._output_channels)) ** 0.5
        recurrent_weights = dict(zip(
            ['layer_' + str(i) for i in range(self._num_layers)],
            [{'w': ivy.variable(ivy.random_uniform(-wlim, wlim, (self._output_channels, 4 * self._output_channels)))}
             for i in range(self._num_layers)]))
        return {'input': input_weights, 'recurrent': recurrent_weights}

    def _forward(self, inputs, v, initial_state=None):
        """
        Perform forward pass of the lstm layer.

        :param inputs: Inputs to process *[batch_shape, t, in]*.
        :type inputs: array
        :param v: the variables for each of the lstm cells, as a container, use internal variables by default.
        :type v: ivy container of parameter arrays, optional
        :param initial_state: 2-tuple of lists of the hidden states h and c for each layer, each of dimension *[batch_shape,out]*.
                        Created internally if None.
        :type initial_state: tuple of list of arrays, optional
        :return: The outputs of the final lstm layer *[batch_shape, t, out]* and the hidden state tuple of lists,
                each of dimension *[batch_shape, out]*
        """
        if initial_state is None:
            initial_state = self.get_initial_state(inputs.shape[:-2])
        h_n_list = list()
        c_n_list = list()
        h_t = inputs
        for h_0, c_0, (_, lstm_input_var), (_, lstm_recurrent_var) in zip(
                initial_state[0], initial_state[1], v.input.items(), v.recurrent.items()):
            h_t, c_n = ivy.lstm_update(h_t, h_0, c_0, lstm_input_var.w, lstm_recurrent_var.w)
            h_n_list.append(h_t[..., -1, :])
            c_n_list.append(c_n)
        if not self._return_sequence:
            h_t = h_t[..., -1, :]
        if not self._return_state:
            return h_t
        return h_t, (h_n_list, c_n_list)
