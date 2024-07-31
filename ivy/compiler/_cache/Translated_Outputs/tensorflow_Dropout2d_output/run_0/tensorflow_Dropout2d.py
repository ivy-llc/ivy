from .tensorflow__DropoutNd import tensorflow__DropoutNd
from .tensorflow__helpers import tensorflow_dropout2d_frnt
from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output


class tensorflow_Dropout2d(tensorflow__DropoutNd):
    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input):
        return tensorflow_dropout2d_frnt(input, self.p, self.training, self.inplace)
