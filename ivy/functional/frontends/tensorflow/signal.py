import ivy
import ivy.functional.frontends.tensorflow as ivy_tf

# hann_window
def hann_window(window_length, periodic=True, dtype=ivy_tf.dtypes.float32, name=None): # ivy dtype is a better choice here
    # you need to either implement it using compositions, or wait for ivy's own version of this function to be implemented
    # more details in the deep dive document https://lets-unify.ai/ivy/deep_dive/16_ivy_frontends.html#ivy-frontends
    pass
