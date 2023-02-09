import ivy


def is_classifier(estimator):
    return ivy.is_classifier(estimator)


def is_regressor(estimator):
    return ivy.is_regressor
