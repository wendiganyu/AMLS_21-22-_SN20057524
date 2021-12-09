import numpy as np

import PreProcessing
import logistic
from RandomForest import RF_Classifier_and_Reducer
from SVM_binary import SVM_binary
from kNN import kNNClassifier
from sklearn import metrics



def MajorVote_Binary(x_train, x_valid, y_train, y_valid):
    # Acquire results from different models.
    _, y_pred_RF, x_train_reduced, x_valid_reduced, _ = RF_Classifier_and_Reducer(x_train, x_valid, y_train, y_valid, x_train)

    # _, y_pred_SVM = SVM_binary(x_train_reduced, x_valid_reduced, y_train, y_valid)
    _, y_pred_kNN = kNNClassifier(x_train, x_valid, y_train, y_valid, k=1)
    _, y_pred_LR = logistic.LR_Classifier(x_train, x_valid, y_train, y_valid)

    y_pred_MV = []
    for i in range(len(y_valid)):
        # Initialize counters
        num_class_0 = 0
        num_class_1 = 0

        # tmp_y_SVM = y_pred_SVM[i]
        tmp_y_kNN = y_pred_kNN[i]
        tmp_y_RF = y_pred_RF[i]
        tmp_y_LR = y_pred_LR[i]

        tmp = [tmp_y_kNN, tmp_y_RF, tmp_y_LR]

        for j in tmp:
            if j == 0:
                num_class_0 += 1
            elif j == 1:
                num_class_1 += 1
            else:
                print("Error! More than to classes!")

        if num_class_0 > num_class_1:
            y_pred_MV.append(0)
        elif num_class_0 < num_class_1:
            y_pred_MV.append(1)
        else:
            y_pred_MV.append(0)

    report = metrics.classification_report(y_valid, y_pred_MV)
    print("Classification report of major Voting combining SVM, kNN and Random Forest:\n " + report)
    accu = metrics.accuracy_score(y_valid, y_pred_MV)

    return accu, y_pred_MV

def k_fold_major_vote(k, X, Y):
    stf_K_fold = StratifiedKFold


if __name__ == '__main__':
    random_state = 55
    x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False, random_state=random_state)
    MajorVote_Binary(x_train, x_valid, y_train, y_valid)