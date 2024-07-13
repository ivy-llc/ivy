from .core import Booster


def train(
    params,
    dtrain,
    dlabel,
    num_boost_round=10,
    *,
    evals=None,
    obj=None,
    feval=None,
    maximize=None,
    early_stopping_rounds=None,
    evals_result=None,
    verbose_eval=True,
    xgb_model=None,
    callbacks=None,
    custom_metric=None,
):
    """Train a booster with given parameters.

    Parameters
    ----------
    params
        Booster params.
    dtrain
        Data to be trained.
    dlabel
        Training labels.
    num_boost_round
        Number of boosting iterations.
    evals
        List of validation sets for which metrics will be evaluated during training.
        Validation metrics will help us track the performance of the model.
    obj
        Custom objective function.
    feval
        Deprecated.
    maximize
        Whether to maximize feval.
    early_stopping_rounds
        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        Requires at least one item in **evals**.
        The method returns the model from the last iteration (not the best one).  Use
        custom callback or model slicing if the best model is desired.
        If there's more than one item in **evals**, the last entry will be used for
        early stopping.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
        If early stopping occurs, the model will have two additional fields:
        ``bst.best_score``, ``bst.best_iteration``.
    evals_result
        This dictionary stores the evaluation results of all the items in watchlist.
    verbose_eval
        Requires at least one item in **evals**.
        If **verbose_eval** is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If **verbose_eval** is an integer then the evaluation metric on the validation
        set is printed at every given **verbose_eval** boosting stage. The last boosting
        stage / the boosting stage found by using **early_stopping_rounds** is also
        printed.
        Example: with ``verbose_eval=4`` and at least one item in **evals**, an
        evaluation metric is printed every 4 boosting stages, instead of every boosting
        stage.
    xgb_model
        Xgb model to be loaded before training (allows training continuation).
    callbacks
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks.
    custom_metric
        Custom metric function.

    Returns
    -------
    Booster : a trained booster model
    """
    # this function creates an instance of Booster and calls its update method
    # to learn model parameters
    # ToDo: add handling for callbacks and write training history

    bst = Booster(params, cache=[dtrain, dlabel], model_file=xgb_model)

    for i in range(num_boost_round):
        bst.update(dtrain, dlabel, iteration=i, fobj=obj)

    return bst
