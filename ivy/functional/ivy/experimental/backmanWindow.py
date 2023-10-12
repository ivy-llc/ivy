import ivy
import numpy as np
def BlackmanWindow(window_length):
    if window_length == 1:
        return ivy.ones(1)
    odd = window_length % 2
    if not odd:
        window_length += 1
    window = 0.42 - 0.5 * ivy.cos(2.0 * np.pi * ivy.arange(window_length) / (window_length - 1)) + \
             0.08 * ivy.cos(4.0 * np.pi * ivy.arange(window_length) / (window_length - 1))
    if not odd:
        window = window[:-1]
    return window

window_length = 10
window = BlackmanWindow(window_length)
print(window)