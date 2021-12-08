from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import os
import pickle as pk

import PreProcessing
import DimensionReduction
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
    # # Do PCA with x_train and x_test.
    # print("I am doing PCA!")
    # x_train, _, Variance, _ = FeatureExtraction.PCAFeatureExtraction(x_train, k)
    # print("Variance after PCA:",Variance)
    # print("x_train after PCA:\n", x_train.shape, x_train)
    #
    # print("Train PCA model done!")
    # # Load the trained PCA model from pca_100.pkl
    # print("I am loading pca model from file")
    # pca_model_name = "pca_" + str(k) + ".pkl"
    # trained_pca_model = pk.load(open(pca_model_name, 'rb'))
    # print("Loading PCA model done! Doing transform of x_test")
    # x_test = trained_pca_model.transform(x_test)
    # print("x_test after PCA:\n", x_test.shape, x_test)

    # ------------------------------------------------------------------------------------------------------------------
    # Construct SVM classification model.

    param_grid = {'C': [0.1, 1, 5, 10, 15, 20], 'gamma': [0.0001], 'kernel': ['rbf']}

    svc = svm.SVC(probability=True)

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
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_valid))
    print(f"The model is {accuracy_score(y_pred, y_valid) * 100}% accurate")


if __name__ == '__main__':
    random_state = 108
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False, random_state=random_state)
    # x_train = PreProcessing.standardization(x_train)
    # x_valid = PreProcessing.standardization(x_valid)
    # k = 2100
    #
    # x_train, x_valid, _, _, _ = DimensionReduction.PCAFeatureExtraction(x_train, x_valid, k,
    # random_state=random_state)

    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False, random_state=random_state)

    # x_train = PreProcessing.standardization(x_train)
    # x_valid = PreProcessing.standardization(x_valid)

    score, x_train, x_valid = RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid,
                                                        random_state=random_state)

    # print(x_train[1])
    # print(x_train_pca_std[1])
    SVM_binary(x_train, x_valid, y_train, y_valid)
