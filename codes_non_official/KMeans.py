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