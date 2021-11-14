from sklearn.decomposition import PCA



def PCAFeatureExtraction(X, k):
    """
    Do the Principle Component Analysis (PCA) method to original input data
    to get the dimensional reduced output, which is smaller and reduces disturbance
    of outliers, with the help of sklearn functions.

    Inputs
        X: dataset;
        k: number of components.

    Return
        pca_data: reduced dataset after PCA. This data varies with the changing of input k value.
        SValue: The singular values corresponding to each of the selected components.
        Variance: The amount of variance explained by each of the selected components.
                It will provide you with the amount of information or variance each principal component holds after projecting the data to a lower dimensional subspace.
        Vcomp: The estimated number of components.
    """

    # Built-in function for PCA,
    # where n_clusters is the number of clusters.
    pca = PCA(n_components=k)

    # Fit the algorithm with dataset
    pca.fit(X)

    pca_data = pca.transform(X)
    SValue = pca.singular_values_
    Variance = pca.explained_variance_ratio_
    Vcomp = pca.components_
    return pca_data, SValue, Variance, Vcomp