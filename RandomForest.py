import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel

import DimensionReduction
import PreProcessing
import pickle as pk



def RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid, random_state):
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
    # Create NB object with a K coefficient
    tree_params = {
        'criterion': 'entropy'
    }

    rf_model_name = "tmp/"+"rf_randomState" + str(random_state) + ".pkl"
    if os.path.exists(rf_model_name):
        print("Yes you are loading rf model from saved pickle file!")
        rf = pk.load(open(rf_model_name, 'rb'))
    else:
        rf = tree.DecisionTreeClassifier(**tree_params, random_state=0)
        rf.fit(x_train, y_train)  # Fit NB model

        # Save the trained PCA model to a file.
        # Set protocol=4 to be able to save large pickle files.
        pk.dump(rf, open(rf_model_name, "wb"), protocol=4)



    y_pred = rf.predict(x_valid)

    report = metrics.classification_report(y_valid, y_pred)
    print("Random Forest classification report:\n " + report)
    score = metrics.accuracy_score(y_valid, y_pred)
    # print("Logistic classifier accuracy: " + str(score))

    # Dimension Reduction

    # -------------------------------------------------------------------------------------------------
    # Check for the proper number of reduced dimensions
    # importance = clf.feature_importances_
    # print(importance)
    # indices = np.argsort(importance)[-70:]  # top 100 features
    # plt.title('Feature Importance')
    # plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    # plt.xlabel('Relative Importance')
    # plt.show()
    # With experiments, reduce to >= 50 features is ok.

    x_train_reduced = SelectFromModel(rf, prefit=True).transform(x_train)
    x_valid_reduced = SelectFromModel(rf, prefit=True).transform(x_valid)

    return score, x_train_reduced, x_valid_reduced


if __name__ == '__main__':
    random_state = 108
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False, random_state=108)

    # x_train = PreProcessing.standardization(x_train)
    # x_valid = PreProcessing.standardization(x_valid)

    score, x_train_reduced, x_valid_reduced = RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid, random_state=random_state)

    print(x_train_reduced.shape)
    print(x_train_reduced)
    print(x_valid_reduced.shape)
    print(x_valid_reduced)
