
# global
#hann_window = ivy.hann_window
#hamming_window = ivy.hamming_window
#blackman_window = ivy.blackman_window
#bartlett_window = ivy.bartlett_window
#kaiser_window = ivy.kaiser_window
#stft = ivy.stft

# local
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

# Tests
# -------   


def test_hann_window():
    x = Tensor([1, 2, 3, 4])
    assert x.hann_window().data == ivy.hann_window([1, 2, 3, 4])


def test_hamming_window():
    x = Tensor([1, 2, 3, 4])
    assert x.hamming_window().data == ivy.hamming_window([1, 2, 3, 4])


def test_blackman_window():
    x = Tensor([1, 2, 3, 4])
    assert x.blackman_window().data == ivy.blackman_window([1, 2, 3, 4])


def test_bartlett_window():
    x = Tensor([1, 2, 3, 4])
    assert x.bartlett_window().data == ivy.bartlett_window([1, 2, 3, 4])


def test_kaiser_window():
    x = Tensor([1, 2, 3, 4])
    assert x.kaiser_window().data == ivy.kaiser_window([1, 2, 3, 4])


def test_stft():
    x = Tensor([1, 2, 3, 4])
    assert x.stft().data == ivy.stft([1, 2, 3, 4])


if __name__ == '__main__':
    test_hann_window()
    test_hamming_window()
    test_blackman_window()
    test_bartlett_window()
    test_kaiser_window()
    test_stft()

# Output
# -------
# test_hann_window()
# test_hamming_window()
# test_blackman_window()
# test_bartlett_window()
# test_kaiser_window()
# test_stft()

# Conclusion
# ------------
# This is a very simple example, but it shows how to extend the ivy API with custom methods.
# The only thing to remember is that the ivy API is not a class, but a module, so you can't
# extend it with class methods. Instead, you have to extend it with module functions.
#
