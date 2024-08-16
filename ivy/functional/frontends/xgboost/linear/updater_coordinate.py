import ivy
from ivy.functional.frontends.xgboost.linear.coordinate_common import (
    get_bias_gradient,
    coordinate_delta_bias,
    update_bias_residual,
    coordinate_delta,
)


def coordinate_updater(gpair, data, lr, weight, n_feat, n_iter, reg_alpha, reg_lambda):
    """Implements one step of coordinate descent. The original optimizer
    implements parallel calculations. The below code is an approximation of the
    original one, but rather than computing the update direction for a single
    parameter at a time using a for loop and cumulative gradients, it does the
    update in parallel by means of matrix-vector multiplications. Given that
    xgboost's updater is non-deterministic, the approximated and original
    implementations converge to pretty the same optima, resulting in metrics'
    values(accuracy, f1-score) differing at a level of 0.001(for separate runs
    metrics may end up being the same).

    Parameters
    ----------
    gpair
        Array of shape (n_samples, 2) holding gradient-hessian pairs.
    data
        Training data of shape (n_samples, n_features).
    lr
        Learning rate.
    weight
        Array of shape (n_features+1, n_output_group) holding feature weights
        and biases.
    n_feat
        Number of features in the training data.
    n_iter
        Number of current iteration.
    reg_alpha
        Denormalized regularization parameter alpha.
    reg_lambda
        Denormalized regularization parameter lambda.

    Returns
    -------
        Updated weights of shape (n_features+1, n_output_group).
    """
    # update biases for all groups
    bias_grad = get_bias_gradient(gpair)
    dbias = lr * coordinate_delta_bias(bias_grad[0], bias_grad[1])
    bias_weights = weight[-1] + dbias

    # upd gradients with bias delta and extract hessians
    grad = update_bias_residual(dbias, gpair)
    hess = ivy.expand_dims(gpair[:, 1], axis=1)

    # don't update where hessian is less than zero
    mask = ivy.where(hess < 0.0, 0.0, 1.0)
    sum_hess = ivy.sum(ivy.square(data) * hess * mask, axis=0, keepdims=True)
    sum_grad = ivy.sum(data * grad * mask, axis=0, keepdims=True)

    # we transpose the arrays to convert (1, n_features) to (n_features, 1)
    dw = lr * coordinate_delta(
        sum_grad.T, sum_hess.T, weight[:-1, :], reg_alpha, reg_lambda
    )
    feature_weights = weight[:-1] + dw

    # faster updates because some backends doesn't support inplace updates
    # speeds up training time because we don't need to create copies implicitly
    return ivy.vstack([feature_weights, bias_weights])
