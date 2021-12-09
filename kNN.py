import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import DimensionReduction
import PreProcessing
from RandomForest import RF_Classifier_and_Reducer


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

    report = metrics.classification_report(y_valid, y_pred)
    print("Random Forest classification report:\n " + report)
    score = metrics.accuracy_score(y_valid, y_pred)
    print("kNN accuracy with number of neighbors k=" + str(k) + ":" + str(score * 100) + "%")
    return score, y_pred


def kNNClassifier_CV(X, Y, k):
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

    scores = cross_val_score(neigh, X, Y, cv=5)
    print("scores of 5-fold validation: ")
    print(scores)

    accu = scores.mean()
    std = scores.std()
    print("%0.2f accuracy with a standard deviation of %0.2f" % (accu, std))

    return accu, std


if __name__ == '__main__':
    # random_state = 108
    # x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False, random_state=108)
    # x_train_pca, x_valid_pca = DimensionReduction.kPCAFeatureExtraction(x_train, x_valid, 100)
    # x_train_pca_std = PreProcessing.standardization(x_train_pca)
    # x_valid_pca_std = PreProcessing.standardization(x_valid_pca)

    # x_train_std = PreProcessing.standardization(x_train)
    # x_valid_std = PreProcessing.standardization(x_valid)
    # score, x_train, x_valid = RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid, random_state=random_state)

    # err = []
    # for k in range(1, 200):
    #     score = kNNClassifier(x_train, x_valid, y_train, y_valid, k)
    #     err.append(score)
    # score = kNNClassifier(x_train, x_valid, y_train, y_valid, k=1)

    # plt.plot(np.arange(1, 200), err)
    # plt.show()
    # score = kNNClassifier(x_train, x_valid, y_train, y_valid)

    # ---------------------------------------------------------------------------------------------------
    X, Y = PreProcessing.gen_X_Y(is_mul=False)
    kNNClassifier_CV(X, Y, k=1)
