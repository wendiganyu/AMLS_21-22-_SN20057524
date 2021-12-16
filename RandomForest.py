import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold

import PreProcessing
from sklearn import tree
import graphviz
from numpy import random

import os

os.environ["PATH"] += os.pathsep + "D:/Graphviz/bin/"


def RF_Classifier(x_train, x_valid, y_train, y_valid, n_estimators):
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

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    rf.fit(x_train, y_train)  # Fit RF model

    # Plot one of the decision tree
    # text_representation = tree.export_text(rf.estimators_[1])
    # print(text_representation)

    # x = random.randint(100)
    # dot_data = tree.export_graphviz(rf.estimators_[x])
    # graph = graphviz.Source(dot_data)
    # graph.render("DecisionTree_"+str(x))

    y_pred = rf.predict(x_valid)

    report = metrics.classification_report(y_valid, y_pred)
    print("Random Forest classification report:\n " + report)
    accu = metrics.accuracy_score(y_valid, y_pred)
    print("Random Forest classification accuracy: " + str(accu))

    return accu, y_pred


if __name__ == '__main__':
    is_mul = True
    stf_K_fold = StratifiedKFold(n_splits=5)
    X, Y = PreProcessing.gen_X_Y(is_mul=is_mul)
    x_test, y_test = PreProcessing.gen_test_X_Y(is_mul=is_mul)
    scores = []
    for train_idx, valid_idx in stf_K_fold.split(X, Y):
        # print("TRAIN:", train_idx, "TEST:", valid_idx)
        x_train, _ = X[train_idx], X[valid_idx]
        y_train, _ = Y[train_idx], Y[valid_idx]

        score, _ = RF_Classifier(x_train, x_test, y_train, y_test, n_estimators=650)
        # RF_Classifier(x_train, x_test, y_train, y_test, n_estimators=650)
        scores.append(score)
    print(scores)
    avg_accu = np.array(scores).mean()
    std = np.array(scores).std()
    #
    print("RF with 5-fold stratified cross validation: %0.5f accuracy with a standard deviation of %0.5f" % (avg_accu, std))

    # x_train, _, y_train, _ = PreProcessing.gen_train_valid_set(is_mul=False)
