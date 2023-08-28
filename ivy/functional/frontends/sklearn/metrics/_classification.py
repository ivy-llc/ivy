import ivy


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    # TODO: implement sample_weight
    # TODO: multi-class
    ret = ivy.equal(y_true, y_pred)
    ret = ret.sum()
    if normalize:
        ret = ret / y_true.shape[0]
    return ret
