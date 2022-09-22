
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend


class Tensor:
    def __init__(self, data):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        self.data = data

    # Instance Methoods #
    # -------------------#

    def Reshape(self, shape, name="Reshape"):
        return tf_frontend.Reshape(self.data, shape, name)

    def add(self, y, name="add"):
        return tf_frontend.add(self.data, y, name)

    def hann_window(self, name="hann_window"):
        return ivy.hann_window(self.data, name)

    def hamming_window(self, name="hamming_window"):
        return ivy.hamming_window(self.data, name)

    def blackman_window(self, name="blackman_window"):
        return ivy.blackman_window(self.data, name)

    def bartlett_window(self, name="bartlett_window"):
        return ivy.bartlett_window(self.data, name)

    def kaiser_window(self, name="kaiser_window"):
        return ivy.kaiser_window(self.data, name)

    def stft(self, name="stft"):
        return ivy.stft(self.data, name)
