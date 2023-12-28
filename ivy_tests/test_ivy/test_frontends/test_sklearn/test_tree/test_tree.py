import ivy.functional.frontends.sklearn as sklearn_frontend


# helper functions
def _get_sklearn_predict(X, y, max_depth):
    ivy_clf = sklearn_frontend.tree.DecisionTreeClassifier(max_depth=max_depth)
    ivy_clf.fit(X, y)
    return ivy_clf.predict
