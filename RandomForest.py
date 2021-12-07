import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import metrics

import DimensionReduction
import PreProcessing


def RF_Classifier(x_train, x_valid, y_train, y_valid):
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
    clf = tree.DecisionTreeClassifier(**tree_params, random_state=0)
    clf.fit(x_train, y_train)  # Fit NB model

    y_pred = clf.predict(x_valid)

    importance = clf.feature_importances_
    indices = np.argsort(importance)[-100:]  # top 100 features
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.xlabel('Relative Importance')

    plt.show()

    report = metrics.classification_report(y_valid, y_pred)
    print("Random Forest classification report:\n " + report)
    score = metrics.accuracy_score(y_valid, y_pred)
    # print("Logistic classifier accuracy: " + str(score))
    return score


if __name__ == '__main__':
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False)
    # x_train_pca, x_valid_pca = DimensionReduction.kPCAFeatureExtraction(x_train, x_valid, 100)
    # x_train_pca_std = PreProcessing.standardization(x_train_pca)
    # x_valid_pca_std = PreProcessing.standardization(x_valid_pca)

    x_train_std = PreProcessing.standardization(x_train)
    x_valid_std = PreProcessing.standardization(x_valid)

    score = RF_Classifier(x_train, x_valid, y_train, y_valid)
