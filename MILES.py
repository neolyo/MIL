# -*- encoding: utf-8 -*-
'''
@File    :   MILES.py
@Time    :   2020/08/29 12:29:00
@Author  :   Xiaoxiao Ma, Jia Wu
@Version :   1.0
@Contact :   xiaoxiao.ma2@hdr.mq.edu.au
@Desc    :   Contact author for details
@Original Paper: MILES: Multiple-instance learning via embedded instance selection
@Paper Link:     https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1717454
'''

# here put the import lib
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linprog
import math
import statistics

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss
import pulp
from pulp import *
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


from utils import load_data, FeatureMapping, evaluation

class MILES():
    '''
    MILES model. 
    Evaluation Steps of this model
    1. Build All instances set - the concept set
    2. Feature mapping - map training bags to a feature space build by the concept set
    3. Classification
    '''

    def __init__(self, sigma2):
        super(MILES, self).__init__()
        self.sigma2 = sigma2
    
    # Evaluation.
    def evaluate(self, bags_features, bags_labels, fold, classifier):
        '''
        bags_features, bags_labels, fold, classifier
        '''
        self.N_fold_CV(bags_features, bags_labels, fold, classifier)
        pass

    def N_fold_CV(self, all_bags, all_bags_labels, fold, classifier):
        
        result = []
        bag_indices = np.array(list(all_bags.keys()))
        kfold = KFold(fold, True, 1)
        
        train_bags = {}
        train_label = {}
        test_bags = {}
        test_label = {}
        fold_index = 1
        
        # Perform N fold tests.
        for train, test in kfold.split(bag_indices):
            
            #print("Performing test on Fold {}:".format(fold_index))
            fold_index += 1
            # Build training bags
            for train_bag_index in bag_indices[train]:
                train_bags.update({train_bag_index:all_bags[train_bag_index]})
                train_label.update({train_bag_index:all_bags_labels[train_bag_index]})

            # Build test bags
            for test_bag_index in bag_indices[test]:
                test_bags.update({test_bag_index:all_bags[test_bag_index]})
                test_label.update({test_bag_index:all_bags_labels[test_bag_index]})
            
            # Perform test on each fold.
            result.append(self.One_fold_CV(train_bags, train_label, test_bags, test_label, classifier))

        auc = []
        f1 = []
        for rec in result:
            auc.append(rec["auc"])
            f1.append(rec["f1"])
        
        avg_auc = statistics.mean(auc)
        stdv_auc = statistics.stdev(auc)
        
        avg_f1 = statistics.mean(f1)
        stdv_f1 = statistics.stdev(f1)
        
        print("The MILES's {} Fold test result on classifier {} is: ".format(fold, classifier))
        print("F1 score: {:.4f} +/- {:.4f}, AUC: {:.4f} +/- {:.4f} ".format(avg_f1, stdv_f1, avg_auc, stdv_auc))

    def One_fold_CV(self, train_bags, train_label, test_bags, test_label, classifier):
        
        # Step1 instance pool.
        # For MILES, the instance pool is constructed using all instances in all training bags.
        instance_pool = []
        for bag_index in train_bags.keys():
            for instance in train_bags[bag_index]:
                instance_pool.append(instance)

        # Step2 bag feature mapping. Training bags feature mapping and test bags feature mapping.
        train_bags_features = FeatureMapping(train_bags, instance_pool, self.sigma2)
        test_bags_features = FeatureMapping(test_bags, instance_pool, self.sigma2)
        
        # Step3 train the classifier and get the test result.
        arr_train_features = []
        arr_train_label = []
        
        for bag in train_bags_features.keys():
            arr_train_features.append(train_bags_features[bag])
            arr_train_label.append(train_label[bag])

        arr_train_features = np.array(arr_train_features)
        arr_train_label = np.array(arr_train_label)
        
        arr_test_features = []
        arr_test_label = []
        
        for bag in test_bags_features.keys():
            arr_test_features.append(test_bags_features[bag])
            arr_test_label.append(test_label[bag])

        arr_test_features = np.array(arr_test_features)
        arr_test_label = np.array(arr_test_label)
        
        if classifier == "svm":
            # SVM classifier
            clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, max_iter=10000)
            clf.fit(arr_train_features, arr_train_label)
            preds = clf.predict(arr_test_features)
            svm_result = evaluation(arr_test_label, preds)
            #for rs in svm_result:
                #print("SVM evaluation {} score is: {}".format(rs,svm_result[rs]))
            
            result = svm_result

        if classifier == "knn":
            # KNN classifier
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(arr_train_features, arr_train_label)
            #knn_score = knn.score(arr_test_features, arr_test_label)
            preds = knn.predict(arr_test_features)
            #print("SVM score:", clf.score(arr_test_features, arr_test_label))
            knn_result = evaluation(arr_test_label, preds)
            #for rs in knn_result:
                #print("KNN evaluation {} score is: {}".format(rs,knn_result[rs]))
            
            result  = knn_result

        if classifier == "nb":
            # Naive Bayes - Gaussian naive bayes classifier
            gnb = GaussianNB()
            gnb.fit(arr_train_features, arr_train_label)
            preds = gnb.predict(arr_test_features)
            nb_result = evaluation(arr_test_label, preds)
            #for rs in nb_result:
                #print("Gaussian Naive Bayes evaluation {} score is: {}".format(rs,nb_result[rs]))

            result = nb_result

        if classifier == "mlp":
            # MLP - DNN
            mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000)
            mlp.fit(arr_train_features, arr_train_label)           
            preds = mlp.predict(arr_test_features)
            mlp_result = evaluation(arr_test_label, preds)
            #for rs in mlp_result:
                #print("MLP evaluation {} score is: {}".format(rs,mlp_result[rs]))
            result = mlp_result
        
        return result