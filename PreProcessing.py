import os
import cv2
import pandas as pd
import numpy as np


# from sklearn.model_selection import train_test_split

def gen_mri_mtx_with_label(data_dir, label_path, is_mul):
    """
    Transform the 3000 raw .jpg images in MRI dataset combined with
    their binary label data indicating the existence of brain tumor or not
    together to a big numpy matrix we could use in program processing.

    Input
        data_dir：location of the dataset directory
        label_path: location of label file of the dataset.
        is_mul: Generate the matrix differently depending whether it will be used in binary or multiple
                classification tasks.
    Return
        mri_mtx: MRI dataset matrix. Each image is represented as
                 a row of 262145 * 1. The first 262144 (512 * 512)
                 elements are data points of image. The last 1 element
                 is its class (0:no_tumor, 1:with_tumor).
    """

    mri_mtx = np.empty((0, 262145), int)
    df = pd.read_csv(label_path)
    df = df.set_index('file_name')

    for fileName in sorted(os.listdir(data_dir)):
        print("This is img " + fileName)
        # Read an image as a matrix.
        full_path = os.path.join(data_dir, fileName)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        # Determine this image's class
        label_val = df.loc[fileName, "label"]
        if label_val == "no_tumor":
            img_class = 0  # 0: this img indicates no tumor, vice versa.
        else:
            if is_mul:
                if label_val == "meningioma_tumor":
                    img_class = 1  # 1 for meningioma tumor
                elif label_val == "glioma_tumor":
                    img_class = 2  # 2 for glioma tumor
                else:
                    img_class = 3  # 3 for pituitary_tumor
            else:
                img_class = 1

        # Convert original image matrix to a row vector appended with its class.
        img_row_vec = img.flatten()
        data_vec = np.append(img_row_vec, img_class)

        # Append this row data vector to mri_matrix.
        mri_mtx = np.append(mri_mtx, [data_vec], axis=0)
    return mri_mtx


def gen_X_Y(is_mul, data_dir="dataset/image", label_path="dataset/label.csv"):
    """
    Transform the 3000 raw .jpg images in MRI dataset combined with
    their binary label data indicating the existence of brain tumor or not
    together to a big numpy matrix we could use in program processing.

    Input
        data_dir：location of the dataset directory
        label_path: location of label file of the dataset.
        is_mul: Generate the results differently depending whether it will be used in binary or multiple
                classification tasks.
    Return
        mri_mtx: MRI dataset matrix. Each image is represented as
                 a row of 262145 * 1. The first 262144 (512 * 512)
                 elements are data points of image. The last 1 element
                 is its class (0:no_tumor, 1:with_tumor).
    """
    if is_mul:
        mtx_file_name = "tmp/MRI_Matrix_Mul.npy"
    else:
        mtx_file_name = "tmp/MRI_Matrix_Binary.npy"

    if os.path.exists(mtx_file_name):
        mri_mtx = np.load(mtx_file_name)
    else:
        mri_mtx = gen_mri_mtx_with_label(data_dir, label_path, is_mul=is_mul)
        os.makedirs('tmp/', exist_ok=True)
        np.save(mtx_file_name, mri_mtx)

    # Split X and Y.
    Y = mri_mtx[:, -1]
    X = np.delete(mri_mtx, 262144, 1)

    return X, Y


def gen_test_X_Y(is_mul, data_dir="test/image", label_path="test/label.csv"):
    if is_mul:
        mtx_file_name = "tmp/test_MRI_Matrix_Mul.npy"
    else:
        mtx_file_name = "tmp/test_MRI_Matrix_Binary.npy"

    if os.path.exists(mtx_file_name):
        mri_mtx = np.load(mtx_file_name)
    else:
        mri_mtx = gen_mri_mtx_with_label(data_dir, label_path, is_mul=is_mul)
        os.makedirs('tmp/', exist_ok=True)
        np.save(mtx_file_name, mri_mtx)

    # Split X and Y.
    Y = mri_mtx[:, -1]
    X = np.delete(mri_mtx, 262144, 1)

    return X, Y


'''
# This function is other way to split the dataset, but it was not used in official implementation.
def gen_train_valid_set(is_mul, random_state=108, test_size=0.2):
    """
    Generate the train set and test set.
    The generated train set and test set are different depending on whether they will be used into binary or multiple
    classification task.
    Some temporary files will be output by the function to store the datasets in the processing.

    Input
        data_dir：location of the dataset directory
        label_path: location of label file of the dataset.
        is_mul: Generate the sets differently depending whether they will be used in binary or multiple
                classification tasks.
        random_state: random_state used in the function sklearn.model_selection.train_test_split.
        test_szie: test_size used in the function sklearn.model_selection.train_test_split.

    Return
        x_train: Preprocessed brain MRI images as inputs to train a model.
                 Each data is of the form of a one-dimentional vector.
        x_valid: Preprocessed brain MRI images to validate the classification accuracy of the trained model.
                    The preprocessing of x_valid set cannot use any information of x_train or y_train.
                    Each data is of the form of a one-dimentional vector.
        y_train: Label information of x_train as inputs to train a model.
        y_valid: Label information of x_valid validate the classification accuracy of the trained model.
    """
    X, Y = gen_X_Y(is_mul=is_mul)

    # Divide train set and test set.
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    return x_train, x_valid, y_train, y_valid
'''

if __name__ == '__main__':
    from pathlib import Path


    class DisplayablePath(object):
        display_filename_prefix_middle = '├──'
        display_filename_prefix_last = '└──'
        display_parent_prefix_middle = '    '
        display_parent_prefix_last = '│   '

        def __init__(self, path, parent_path, is_last):
            self.path = Path(str(path))
            self.parent = parent_path
            self.is_last = is_last
            if self.parent:
                self.depth = self.parent.depth + 1
            else:
                self.depth = 0

        @property
        def displayname(self):
            if self.path.is_dir():
                return self.path.name + '/'
            return self.path.name

        @classmethod
        def make_tree(cls, root, parent=None, is_last=False, criteria=None):
            root = Path(str(root))
            criteria = criteria or cls._default_criteria

            displayable_root = cls(root, parent, is_last)
            yield displayable_root

            children = sorted(list(path
                                   for path in root.iterdir()
                                   if criteria(path)),
                              key=lambda s: str(s).lower())
            count = 1
            for path in children:
                is_last = count == len(children)
                if path.is_dir():
                    yield from cls.make_tree(path,
                                             parent=displayable_root,
                                             is_last=is_last,
                                             criteria=criteria)
                else:
                    yield cls(path, displayable_root, is_last)
                count += 1

        @classmethod
        def _default_criteria(cls, path):
            return True

        @property
        def displayname(self):
            if self.path.is_dir():
                return self.path.name + '/'
            return self.path.name

        def displayable(self):
            if self.parent is None:
                return self.displayname

            _filename_prefix = (self.display_filename_prefix_last
                                if self.is_last
                                else self.display_filename_prefix_middle)

            parts = ['{!s} {!s}'.format(_filename_prefix,
                                        self.displayname)]

            parent = self.parent
            while parent and parent.parent is not None:
                parts.append(self.display_parent_prefix_middle
                             if parent.is_last
                             else self.display_parent_prefix_last)
                parent = parent.parent

            return ''.join(reversed(parts))


    paths = DisplayablePath.make_tree(Path('./'))
    for path in paths:
        print(path.displayable())