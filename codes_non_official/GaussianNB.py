import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import DimensionReduction
import PreProcessing


def NBClassifier(x_train, x_valid, y_train, y_valid):
    """
    First take the training set, and valid or test set as inputs.
    Then classify them binary with NB. Print the information related to classification accuracy.

    Inputs:
        x_train: Preprocessed brain MRI images as inputs to train a model.
        y_train: Label information of x_train as inputs to train a model.
        x_valid_test: Preprocessed brain MRI images to validate or test the classification accuracy of the trained model.
                    The preprocessing of valid or test sets cannot use any information of x_train or y_train.
        y_valid: Label information of valid or test sets to calculate the classification accuracy of the trained model.

    Outputs:
        accu: Accuracy of the model on valid or test set.
        y_pred: Predicted labels on valid or test set.
    """
    # Create NB object with a K coefficient
    nb_clf = GaussianNB()
    nb_clf.fit(x_train, y_train)  # Fit NB model

    y_pred = nb_clf.predict(x_valid)

    accu = metrics.accuracy_score(y_valid, y_pred)
    print("Gaussian Naive Bayes classifier accuracy: " + str(accu))
    return accu, y_pred


if __name__ == '__main__':
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_valid_set(is_mul=False)
    # x_train_pca, x_valid_pca = DimensionReduction.kPCAFeatureExtraction(x_train, x_valid, 100)
    # x_train_pca_std = PreProcessing.standardization(x_train_pca)
    # x_valid_pca_std = PreProcessing.standardization(x_valid_pca)


    x_train_std = PreProcessing.standardization(x_train)
    x_valid_std = PreProcessing.standardization(x_valid)

    score = NBClassifier(x_train, x_valid, y_train, y_valid)