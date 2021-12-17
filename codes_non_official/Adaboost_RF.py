import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import DimensionReduction
import PreProcessing


def Adaboost_Classifier(x_train, x_valid, y_train, y_valid):
    """
    First take the training set, and valid or test set as inputs.
    Then classify them binary with Adaboost, with base classifier decision tree.
    Print the information related to classification accuracy.

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

    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=2000
    )
    clf.fit(x_train, y_train)  # Fit NB model

    y_pred = clf.predict(x_valid)

    report = metrics.classification_report(y_valid, y_pred)
    print("Adaboost Random Forest classification report:\n " + report)
    accu = metrics.accuracy_score(y_valid, y_pred)

    return accu, y_pred


if __name__ == '__main__':
    # --------------------------------------------------------------------------------------------------------
    # Create Stratified K-Fold model.
    stf_K_fold = StratifiedKFold(n_splits=5)
    X, Y = PreProcessing.gen_X_Y(is_mul=False)

    scores = []
    for train_idx, valid_idx in stf_K_fold.split(X, Y):
        print("TRAIN:", train_idx, "TEST:", valid_idx)
        x_train, x_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = Y[train_idx], Y[valid_idx]

        x_train, x_valid, _, _, _ = DimensionReduction.PCAFeatureExtraction(x_train, x_valid, k=800)
        accu = Adaboost_Classifier(x_train, x_valid, y_train, y_valid)

        scores.append(accu)

    print(str(scores))

    avg_accu = np.array(scores).mean()
    std = np.array(scores).std()

    print("average %0.5f accuracy with a standard deviation of %0.5f" % (avg_accu, std))
