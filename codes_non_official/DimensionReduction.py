import os.path

from sklearn.decomposition import PCA
import pickle as pk

import PreProcessing


def PCAFeatureExtraction(x_train, x_valid, k):
    """
    Do the Principle Component Analysis (PCA) method to original input data
    to get the dimensional reduced output, which is smaller and reduces disturbance
    of outliers, with the help of sklearn functions.
    Note: PCA model can only trained with x_train first. Then use the already-trained model
            to reduce x_valid. Doing this way can avoid using x_valid information
            in training process.

    Inputs
        x_train: training set without label information.
        x_valid: valid set without label information.
        k: number of components.


    Return
        x_train_pca: reduced training set after PCA. This data varies with the changing of input k value.
        x_valid_pca: reduced valid set after PCA. This data varies with the changing of input k value.
        SValue: The singular values corresponding to each of the selected components.
        Variance: The amount of variance explained by each of the selected components.
                It will provide you with the amount of information or variance each principal component holds after projecting the data to a lower dimensional subspace.
        Vcomp: The estimated number of components.
    """

    # Built-in function for PCA,
    # where n_clusters is the number of clusters.

    # Check if there are already-saved pca models to load.


    pca = PCA(n_components=k)
    # Fit the algorithm with dataset
    pca.fit(x_train)

    print("PCA train finish.")


    x_train_pca = pca.transform(x_train)
    x_valid_pca = pca.transform(x_valid)
    SValue = pca.singular_values_
    Variance = pca.explained_variance_ratio_
    Vcomp = pca.components_
    return x_train_pca, x_valid_pca, SValue, Variance, Vcomp

if __name__ == '__main__':
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_valid_set(is_mul=False)
    x_train_pca, x_valid_pca, SValue, Variance, Vcomp = PCAFeatureExtraction(x_train, x_valid, 100)
    print(x_train_pca.shape)
    print(x_valid_pca.shape)
