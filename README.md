# AMLS-I_Final_Project
This is my final project of Applied Machine Learning Systems course I, which does the Brain Tumor MRI images' binary classification and multi-class classification tasks. I used Random Forest for binary task. And with a stratified 5-fold cross validation strategy, the average valid and test accuracies are 95.6% and 97.8% separately. Then, I used a deep residual CNN for both binary and multi-class task, and reached the valid and test accuracies of 99.2%, 99.3% for binary, and 96%, 98% for multi-class task.

# Project Structure
/AMLS_21_22-_SN20057524  
├── CNN_Residual_Structure.py <font color=yellow>Code file to train and validate my deep residual CNN</font>  
├── codes_non_official/  <font color=yellow>You don't need to check the folder unless interested. Unofficial codes that are other models I did experiments with, but not used in my report.</font>  
│   ├── Adaboost_RF.py  
│   ├── DimensionReduction.py  
│   ├── GaussianNB.py  
│   ├── KMeans.py  
│   ├── kNN.py  
│   ├── logistic.py    
│   ├── MajorVote_Binary.py  
│   └── SVM_binary.py  
├── dataset/  <font color=yellow>Dataset of 3000 MRI images and their labels.</font>    
│   ├── image/  
│   └── label.csv  
├── model_states_tmp/  <font color=yellow>The folder to store the state files for each epoch of my CNN model in training process. </font>  
├── PreProcessing.py  <font color=yellow>Python file to concat and convert the dataset of images to a numpy matrix of 3000*262145, and test set to a matrix of 200*262145. And functions to separate and extract data and label information. </font>    
├── RandomForest.py   <font color=yellow>Implementation of my RF model.</font>  
├── README.md  
├── requirements.txt  <font color=yellow>Package requirements of the project.</font>  
├── test/  <font color=yellow>Test set containing 200 MRI images and their labels.</font>  
│   ├── image/  
│   └── label.csv  
├── TestCNN.py  <font color=yellow>Code file to evaluate the performance of my deep residual CNN on test set.</font>   
└── tmp/  <font color=yellow>Folder to store temporary files to accelerate program running and to store some images generated in CNN training.</font>   

 <font color=red>The folders dataset/ and test/ aren't contained under this repository, you need to download them manually.</font>   

<font color=red>The folders model_states_tmp/ and tmp/ aren't contained here either, they will be generated automatically when training the models.</font>

## How to run
See the demonstration about how to run the project is easy and straight: [Demonstration on Colab](https://colab.research.google.com/drive/1DcNPJ3uavPhXdcGJ9YMAqlT-YprJv_vm?usp=sharing)    

First set up the Python environment with required packages. I use the Python version of 3.7.9.

Then, download the dataset: [Dataset](https://drive.google.com/file/d/1Wt7C6SnXx-xloWgScR4S0xHuZ9Y5OJ3E/view) and test set: [TestSet](https://drive.google.com/file/d/1LS_C_4_iOeqOyEoWPPoksrk8lqdBKagB/view), unzip them and put the folders under this project directory.

Then, if you want to train and measure the RF model on valid set, run the command under this project directory:
```
python RandomForest.py
```
Or if you want to train and measure RF on test set, run:
```
python RandomForest.py --isTest
```

To train and measure deep residual CNN model on valid set for binary task, run:
```
python CNN_Residual_Structure.py --epochNum=200
```
where you can change the epochNum to set the epochs to train the model.

Then, for multi-class task:
```
python CNN_Residual_Structure.py --epochNum=200 --isMul
```

To measure the trained out CNN model on test set, first you need to get the model state files by training CNN, or downloading them from my google drive: binary model state files: [Binary task model states files](https://drive.google.com/drive/folders/1-5rumUg_JV_Mr5iGQGU5or3GG-0YNcvZ?usp=sharing), and multi-class model state files: [Multi-class task model states files](https://drive.google.com/drive/folders/1-3LnLBo-DBd_XzkcErsNalwATATSL3-z?usp=sharing)

Then, after acquiring the state file, remember its PATH, and run:
```
python TestCNN.py --PATH=<your state file path> # Test for binary task
python TestCNN.py --PATH=<your state file path> --isMul # Test for multi-class task

```


### Prerequisites

See requirements.txt