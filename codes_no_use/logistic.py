from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import PreProcessing


def LR_Classifier(x_train, x_valid, y_train, y_valid):
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
    clf = LogisticRegression(max_iter=10000, random_state=0)
    clf.fit(x_train, y_train)  # Fit NB model

    y_pred = clf.predict(x_valid)

    score = metrics.accuracy_score(y_valid, y_pred)
    print("Logistic classifier accuracy: " + str(score))
    return score, y_pred


if __name__ == '__main__':
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False)
    # x_train_pca, x_valid_pca = DimensionReduction.kPCAFeatureExtraction(x_train, x_valid, 100)
    # x_train_pca_std = PreProcessing.standardization(x_train_pca)
    # x_valid_pca_std = PreProcessing.standardization(x_valid_pca)


    x_train_std = PreProcessing.standardization(x_train)
    x_valid_std = PreProcessing.standardization(x_valid)

    score = LR_Classifier(x_train, x_valid, y_train, y_valid)
