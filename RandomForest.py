import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import PreProcessing
import argparse

'''
#Optional packages for plotting trees.
import graphviz
from numpy import random
from matplotlib import pyplot as plt
import os
'''


# Below command is used to fix the bug when generating visualized tree in RF.
# os.environ["PATH"] += os.pathsep + "D:/Graphviz/bin/"


def RF_Classifier(x_train, x_valid_test, y_train, y_valid_test, n_estimators):
    """
    First take the training set, and valid or test set as inputs.
    Then classify them binary with RF. Print the information related to classification accuracy.

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
    # Create RF classifier.
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    rf.fit(x_train, y_train)  # Fit RF model

    # Plot one of the decision tree
    # text format
    # text_representation = tree.export_text(rf.estimators_[1])
    # print(text_representation)

    # image format
    # seed = random.randint(100)
    # dot_data = tree.export_graphviz(rf.estimators_[x])
    # graph = graphviz.Source(dot_data)
    # graph.render("DecisionTree_"+str(seed))

    # Make prediction.
    y_pred = rf.predict(x_valid_test)

    # Print results.
    report = metrics.classification_report(y_valid_test, y_pred)
    print("Random Forest classification report:\n " + report)
    accu = metrics.accuracy_score(y_valid_test, y_pred)
    print("Random Forest classification accuracy: " + str(accu))

    return accu, y_pred


if __name__ == '__main__':
    # Get params from command lines.
    p = argparse.ArgumentParser()
    p.add_argument("--isTest", default=False, action="store_true")
    args = p.parse_args()

    # --------------------------------------------------------------------------------------------------
    is_mul = False
    stf_K_fold = StratifiedKFold(n_splits=5)
    X, Y = PreProcessing.gen_X_Y(is_mul=is_mul)
    if args.isTest:
        # Run the RF classifier on test set.
        x_test, y_test = PreProcessing.gen_test_X_Y(is_mul=is_mul)

    scores = []
    for train_idx, valid_idx in stf_K_fold.split(X, Y):
        # print("TRAIN:", train_idx, "TEST:", valid_idx)
        x_train, x_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = Y[train_idx], Y[valid_idx]

        if args.isTest:
            # Train with training set and predict on test set.
            score, _ = RF_Classifier(x_train, x_test, y_train, y_test, n_estimators=650)
        else:
            # Train with training set and predict on valid sets.
            score, _ = RF_Classifier(x_train, x_valid, y_train, y_valid, n_estimators=650)

        scores.append(score)

    print("Accuracies of 5 RF classifiers using stratified 5-fold: "+scores)
    avg_accu = np.array(scores).mean()
    std = np.array(scores).std()

    print("RF with 5-fold stratified cross validation: %0.5f average accuracy with a standard deviation of %0.5f" % (
        str(avg_accu), str(std)))



