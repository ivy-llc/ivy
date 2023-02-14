def k_means(
    X,
    n_clusters,
    *,
    sample_weight=None,
    init="k-means++",
    n_init="warn",
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
):
    return ivy.k_means(x)