from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
import os
import pickle as pk
from sklearn.metrics import accuracy_score
import PreProcessing
import DimensionReduction
import matplotlib.pyplot as plt
from RandomForest import RF_Classifier_and_Reducer


def SVM_binary(x_train, x_valid, y_train, y_valid):
    """
    First take the train data set and valid data set as inputs.
    Then classify them binary with SVM. Print the information related to classification accuracy.

    Inputs:
        x_train: Preprocessed brain MRI images as inputs to train a model.
        y_train: Label information of x_train as inputs to train a model.
        x_valid: Preprocessed brain MRI images to validate the classification accuracy of the trained model.
                    The preprocessing of x_valid set cannot use any information of x_train or y_train.
        y_valid: Label information of x_valid validate the classification accuracy of the trained model.

    """
    # ------------------------------------------------------------------------------------------------------------------
    # Construct SVM classification model.

    param_grid = {'C': [1, 2, 3, 4, 5], 'gamma': [0.0001], 'kernel': ['rbf']}

    svc = svm.SVC()

    # param_grad combined with GridSearchCV helps loop through predefined hyperparameters and fit the model.
    # After finished, we can select the best parameters from the listed hyperparameters.
    svc_model = GridSearchCV(svc, param_grid)

    # ------------------------------------------------------------------------------------------------------------------
    # Train the model
    print("I am training SVC!")
    svc_model.fit(x_train, y_train)
    # model.best_params_ contains the best parameters obtained from GridSearchCV
    print(svc_model.best_params_)
    print(svc_model.best_score_)

    # ------------------------------------------------------------------------------------------------------------------
    # Test the model.
    print("Train SVC done! Doing testing!")
    y_pred = svc_model.predict(x_valid)
    report = classification_report(y_valid, y_pred)
    print("SVM classification report:\n " + report)
    accu = accuracy_score(y_valid, y_pred)
    print(f"The model is {accu * 100}% accurate")

    return accu, y_pred


def SVM_binary_CV(X, Y):
    """
    First take the train data set and valid data set as inputs.
    Then classify them binary with SVM. Print the information related to classification accuracy.

    Inputs:
        x_train: Preprocessed brain MRI images as inputs to train a model.
        y_train: Label information of x_train as inputs to train a model.
        x_valid: Preprocessed brain MRI images to validate the classification accuracy of the trained model.
                    The preprocessing of x_valid set cannot use any information of x_train or y_train.
        y_valid: Label information of x_valid validate the classification accuracy of the trained model.

    """

    # Construct SVM classification model.

    svc = svm.SVC(kernel='rbf', C=5, gamma=0.001)
    scores = cross_val_score(svc, X, Y, cv=5)

    accu = scores.mean()
    std = scores.std()
    print("%0.2f accuracy with a standard deviation of %0.2f" % (accu, std))

    return accu, std


if __name__ == '__main__':
    random_state = 108
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False, random_state=random_state)
    #
    #
    # score, _, x_train, x_valid = RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid,
    #                                                        random_state=random_state)
    #
    # SVM_binary(x_train, x_valid, y_train, y_valid)

    # err = []
    # for k in [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
    #     x_train_reduce, x_valid_reduce, _, _, _ = DimensionReduction.PCAFeatureExtraction(x_train, x_valid, k,
    #     random_state=random_state)
    #     accu, _ = SVM_binary(x_train_reduce, x_valid_reduce, y_train, y_valid)
    #     err.append(accu)
    #
    # plt.plot(np.arange(1,14 ), err)
    # plt.show()

    # -------------------------------------------------------------------------------------------------
    X, Y = PreProcessing.gen_X_Y(is_mul=False)
    score, _, x_train, x_valid, X_reduced = RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid,X,
                                                            random_state=random_state)
    SVM_binary_CV(X_reduced,Y)