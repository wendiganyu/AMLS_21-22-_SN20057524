"""
Main Python file which calls other modules in this directory to realise a whole process from loading original dataset
to outputting the classification accuracy.
"""
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import os
import pickle as pk

import PreProcessing
import FeatureExtraction

data_dir = "dataset/image"  # Path of dataset directory
label_path = "dataset/label.csv"  # Path of dataset's label file

# Path of the .npy file which saves the dataset and its binary labels' information as a matrix.
original_mtx_path = "MRI_Matrix.npy"


def pca_SVM_binary(X, Y, k):
    """
    First do feature extraction of the input dataset using PCA.
    Then classify them binary with SVM.

    Inputs:
        X: brain MRI dataset as the form of a numpy matrix 3000 * 262144.
                Each row represents an image data point.
        Y: label information of the dataset. 0-no tumor. 1-tumor.
        k: number of components of PCA.

    Return:

    """
    # ------------------------------------------------------------------------------------------------------------------
    # Split input data.
    print("I am splitting the data!")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=3)

    print("x_train before PCA:\n", x_train.shape)
    print("y_train:\n", y_train.shape)

    # ------------------------------------------------------------------------------------------------------------------
    # Do PCA with x_train and x_test.
    print("I am doing PCA!")
    x_train, _, Variance, _ = FeatureExtraction.PCAFeatureExtraction(x_train, k)
    print("Variance after PCA:",Variance)
    print("x_train after PCA:\n", x_train.shape, x_train)

    print("Train PCA model done!")
    # Load the trained PCA model from pca_100.pkl
    print("I am loading pca model from file")
    pca_model_name = "pca_" + str(k) + ".pkl"
    trained_pca_model = pk.load(open(pca_model_name, 'rb'))
    print("Loading PCA model done! Doing transform of x_test")
    x_test = trained_pca_model.transform(x_test)
    print("x_test after PCA:\n", x_test.shape, x_test)

    # ------------------------------------------------------------------------------------------------------------------
    # Construct SVM classification model.

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}

    svc = svm.SVC(probability=True)

    # param_grad combined with GridSearchCV helps loop through predefined hyperparameters and fit the model.
    # After finished, we can select the best parameters from the listed hyperparameters.
    svc_model = GridSearchCV(svc, param_grid)

    # ------------------------------------------------------------------------------------------------------------------
    # Train the model
    print("I am training SVC!")
    svc_model.fit(x_train, y_train)
    # model.best_params_ contains the best parameters obtained from GridSearchCV

    # ------------------------------------------------------------------------------------------------------------------
    # Test the model.
    print("Train SVC done! Doing testing!")
    y_pred = svc_model.predict(x_test)
    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")


if __name__ == "__main__":

    # Check if the data matrix already saved as file.
    if os.path.exists(original_mtx_path):
        mri_mtx = np.load(original_mtx_path)
    else:
        mri_mtx = PreProcessing.gen_mri_mtx_binary_label(data_dir, label_path)
        np.save("MRI_Matrix.npy", mri_mtx)
    # print(mri_mtx.shape)

    # Extract data part from the whole matrix.
    X = np.delete(mri_mtx, 262144, 1)

    # Perform data standardization
    # Check if the data matrix already saved as file.

    standard_data_file_name = "Data_After_Standardization.npy"
    if os.path.exists(standard_data_file_name):
        X = np.load(standard_data_file_name)
    else:
        X = PreProcessing.standardization(X)
        np.save(standard_data_file_name, X)
    print(X.shape)

    # Extract label part from the whole matrix.
    Y = mri_mtx[:, -1]

    # Set the number of k components in PCA.
    k = 2400

    pca_SVM_binary(X, Y, k)


