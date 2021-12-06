import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import DimensionReduction
import PreProcessing


def kNNClassifier(x_train, x_valid, y_train, y_valid, k):
    """
    First take the train data set and valid data set as inputs.
    Then classify them binary with kNN. Print the information related to classification accuracy.

    Inputs:
        x_train: Preprocessed brain MRI images as inputs to train a model.
        y_train: Label information of x_train as inputs to train a model.
        x_valid: Preprocessed brain MRI images to validate the classification accuracy of the trained model.
                    The preprocessing of x_valid set cannot use any information of x_train or y_train.
        y_valid: Label information of x_valid validate the classification accuracy of the trained model.
        k: Number of neighbors.

    """
    # Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)  # Fit KNN model

    y_pred = neigh.predict(x_valid)

    score = metrics.accuracy_score(y_valid, y_pred)
    print("kNN accuracy with number of neighbors k=" + str(k) + ":" + str(score))
    return score


if __name__ == '__main__':
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False)
    # x_train_pca, x_valid_pca = DimensionReduction.kPCAFeatureExtraction(x_train, x_valid, 100)
    # x_train_pca_std = PreProcessing.standardization(x_train_pca)
    # x_valid_pca_std = PreProcessing.standardization(x_valid_pca)


    x_train_std = PreProcessing.standardization(x_train)
    x_valid_std = PreProcessing.standardization(x_valid)
    err = []
    for k in range(1, 36):
        score = kNNClassifier(x_train, x_valid, y_train, y_valid, k)
        err.append(score)

    plt.plot(np.arange(1, 36), err)
    plt.show()
    # score = kNNClassifier(x_train, x_valid, y_train, y_valid)
