# MIL - Multi-Instance Learning Models
In this Repository, we have re-implemented the three most representative MIL models - MILIS, MILES and MILDM. The correpsonding paper and details of those models can be found in the header of .py files.

For Simplicity, we use the forementioned three models to generate bag features and then evaluate their performance using 4 classical classifiers, namely, SVM, KNN, Naïve Bayes and MLP.

## 3rd Party package dependency
Those models are implemented in Python 3.7 with Pandas, Numpy, Scipy and SKlearn packages.

## Running & Evaluation
This implementation could parse parameter from the command line and you can use this command to run a simple test on MILES model.

     python main.py --model MILES

Specific Options

| Parameter    | Data Type | Default Value | Options             | Description                                         |
| ------------ | --------- | ------------- | ------------------- | --------------------------------------------------- |
| --dataset    | String    | Musk1         |                     | name of dataset                                     |
| --model      | String    | MILES         | MILES, MILIS, MILDM | name of model                                       |
| --classifier | String    | knn           | svm, knn, nb, mlp   | name of classifier                                  |
| --sigma2     | int       | 800000        |                     | the sigma parameter                                 |
| --fold       | int       | 10            |                     | the fold of cross validation                        |
| --glm        | bool      | True          | True, False         | global or local instance selection                  |
| --pol        | bool      | False         | True, False         | select instance from all bags or positive bags only |

If you find those models are useful for your research, please consider citing them.

