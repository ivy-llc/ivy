import ivy.tracer.globals as glob


def _set_jax_module_mode(ivy_module, kwargs):
    """
    Set the train mode of the graph according to kwargs which
    are usually used in flax/haiku to define the train mode
    """
    if (
        hasattr(ivy_module._module_graph, "_is_trainable_module")
        and ivy_module._module_graph._is_trainable_module
        and ivy_module._module_graph._traced_train_modes == "all"
    ):
        kwargs_ = dict(kwargs)

        if not any([train_kwarg in kwargs_ for train_kwarg in glob.TRAIN_KWARGS]):
            # default to eval mode if no train arguments are found
            ivy_module._module_graph._eval()
        else:
            # find the training argument and set the graph mode accordingly
            for train_kwarg in glob.TRAIN_KWARGS:
                if train_kwarg in kwargs_:
                    (
                        ivy_module._module_graph._train()
                        if kwargs_[train_kwarg]
                        else ivy_module._module_graph._eval()
                    )
                    break  # break the loop once the mode has been set by a kwarg
