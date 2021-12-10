import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel

import DimensionReduction
import PreProcessing
import pickle as pk


def KMeans_Classifier(X, Y, k):
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

    # the bulit-in function for K-means,
    # where n_clusters is the number of clusters.
    kmeans = KMeans(n_clusters=k)

    # fit the algorithm with dataset
    kmeans.fit(X)

    # predict after fit
    y_kmeans = kmeans.predict(X)

    # get the centers after fit
    centers = kmeans.cluster_centers_

    report = metrics.classification_report(Y, y_kmeans)
    print("Kmeans classification report:\n " + report)
    score = metrics.accuracy_score(Y, y_kmeans)

    return score, centers


if __name__ == '__main__':
    mtx_file_name = "tmp/MRI_Matrix_Binary.npy"
    mri_mtx = np.load(mtx_file_name)
    Y = mri_mtx[:, -1]
    mri_mtx = np.delete(mri_mtx, 262144, 1)

    mri_mtx = PreProcessing.standardization(mri_mtx)

    score, x_train, x_valid = RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid,
                                                        random_state=random_state)

    # x_train = PreProcessing.standardization(x_train)
    # x_valid = PreProcessing.standardization(x_valid)

    score, centers = KMeans_Classifier(mri_mtx, Y, k=2)
