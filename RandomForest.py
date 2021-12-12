import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold

import PreProcessing
from sklearn import tree
import graphviz

import os

os.environ["PATH"] += os.pathsep + "D:/Graphviz/bin/"


def RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid):
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
        'criterion': 'entropy',
        'random_state': 0
    }

    # rf_model_name = "tmp/" + "rf_randomState" + str(random_state) + ".pkl"
    # if os.path.exists(rf_model_name):
    #     print("Yes you are loading rf model from saved pickle file!")
    #     rf = pk.load(open(rf_model_name, 'rb'))
    # else:
    rf = RandomForestClassifier(**tree_params)
    rf.fit(x_train, y_train)  # Fit NB model

    # Save the trained PCA model to a file.
    # Set protocol=4 to be able to save large pickle files.
    # pk.dump(rf, open(rf_model_name, "wb"), protocol=4)

    y_pred = rf.predict(x_valid)

    report = metrics.classification_report(y_valid, y_pred)
    print("Random Forest classification report:\n " + report)
    accu = metrics.accuracy_score(y_valid, y_pred)
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
    # X_reduced = SelectFromModel(rf, prefit=True).transform(X)

    return accu, y_pred, x_train_reduced, x_valid_reduced


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
    # Create NB object with a K coefficient
    tree_params = {
        # 'criterion': 'entropy',
        'criterion': 'gini'
        # 'random_state': 110
    }

    # rf_model_name = "tmp/" + "rf_randomState" + str(random_state) + ".pkl"
    # if os.path.exists(rf_model_name):
    #     print("Yes you are loading rf model from saved pickle file!")
    #     rf = pk.load(open(rf_model_name, 'rb'))
    # else:
    rf = RandomForestClassifier(n_estimators=n_estimators, **tree_params)
    rf.fit(x_train, y_train)  # Fit NB model

    # Plot one of the decision tree
    # text_representation = tree.export_text(rf.estimators_[1])
    # print(text_representation)
    # dot_data = tree.export_graphviz(rf.estimators_[1])
    # graph = graphviz.Source(dot_data)
    # graph.render("abc")
    # tree.plot_tree(rf.estimators_[1])

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

    for train_idx, valid_idx in stf_K_fold.split(X, Y):
        # print("TRAIN:", train_idx, "TEST:", valid_idx)
        x_train, x_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = Y[train_idx], Y[valid_idx]
        RF_Classifier(x_train, x_valid, y_train, y_valid, n_estimators=650)
        # RF_Classifier(x_train, x_test, y_train, y_test, n_estimators=650)

    # x_train, _, y_train, _ = PreProcessing.gen_train_valid_set(is_mul=False)

    # --------------------------------------------------------------------------------------------------------
    '''
    # Create Stratified K-Fold model.
    stf_K_fold = StratifiedKFold(n_splits=5)
    X, Y = PreProcessing.gen_X_Y(is_mul=False)
    #
    scores = []
    for train_idx, valid_idx in stf_K_fold.split(X, Y):
        # print("TRAIN:", train_idx, "TEST:", valid_idx)
        x_train, x_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = Y[train_idx], Y[valid_idx]
        # x_train = PreProcessing.BinaryImage(x_train)
        # x_valid = PreProcessing.BinaryImage(x_valid)
        #--------------------------------------------

        # scores = []
        # n_estimators = 600-800 is good
        # tmp_list = [ 600, 650, 700, 750, 800]
        # for n in tmp_list:
        #     accu, _ = RF_Classifier(x_train, x_valid, y_train, y_valid, n_estimators=n)
        #
        #     err.append(accu)
        #
        # plt.plot(np.arange(1, len(tmp_list)+1), err)
        # plt.show()
        #--------------------------------------------
    #     # x_train = x_train - np.mean(x_train)
    #     # x_valid = x_valid - np.mean(x_valid)
    #
    #     # x_train, x_valid, _, _, _ = DimensionReduction.PCAFeatureExtraction(x_train, x_valid, 800)
    #
        accu, _, rf = RF_Classifier(x_train, x_valid, y_train, y_valid, n_estimators=650)
    #
        scores.append(accu)

        visualise_tree(rf.estimators_[0])
    #
    print(scores)
    avg_accu = np.array(scores).mean()
    std = np.array(scores).std()
    #
    print("RF with 5-fold stratified cross validation: %0.5f accuracy with a standard deviation of %0.5f" % (avg_accu, std))

    '''
