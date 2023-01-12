import tensorflow as tf

# def if_exp(cond, if_true, if_false, expr_repr):
#   def true_fn():
#     return if_true()
#
#   def false_fn():
#     return if_false()
#
#   return tf.cond(cond, true_fn, false_fn)


def if_else(cond, body_fn, orelse_fn, vars):
    return tf.cond(cond, lambda: body_fn(*vars), lambda: orelse_fn(*vars))


def while_loop(test_fn, body_fn, vars):
    def body_fn_wrapper(*loop_vars):
        return body_fn(*loop_vars)

    def test_fn_wrapper(*loop_vars):
        return test_fn(*loop_vars)

    return tf.while_loop(test_fn_wrapper, body_fn_wrapper, vars)
