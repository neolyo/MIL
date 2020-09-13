# MIL - Multi-Instance Learning Models
In this Repository, we have re-implemented the three most representative MIL models - MILIS, MILES and MILDM. The correpsonding paper and details of those models can be found in the header of .py files.

For Simplicity, we use the forementioned three models to generate bag features and then evaluate their performance using 4 classical classifiers, namely, SVM, KNN, Naïve Bayes and MLP.

## 3rd Party package dependency
Those models are implemented in Python 3.7 with Pandas, Numpy, Scipy and SKlearn packages.

## Running & Evaluation
This implementation could parse parameter from the command line and you can use this command to run a simple test on MILES model.

>> python main.py


