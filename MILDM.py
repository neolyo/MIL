# -*- encoding: utf-8 -*-
'''
@File    :   MILDM.py
@Time    :   2020/08/29 12:29:15
@Author  :   Xiaoxiao Ma, Jia Wu
@Version :   1.0
@Contact :   xiaoxiao.ma2@hdr.mq.edu.au
@Desc    :   Contact author for details
@Original Paper: Multi-Instance Learning with Discriminative Bag Mapping
@Paper Link:     https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8242668
'''

# here put the import lib
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linprog
import scipy.sparse as sp
from scipy.special import perm
import math
import statistics
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn import metrics
from sklearn.metrics import hinge_loss
from sklearn.metrics.pairwise import manhattan_distances

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from sklearn import datasets
from sklearn.model_selection import KFold

from utils import FeatureMapping, evaluation, cal_each_feature

class MILDM():
    '''
    MILIS model. This model select a representative instance from each bag and then map each bag to a feature space that is measured
    by the similarity between the bag and the selected instances.
    
    Instance selection categories:
    1. From all training bags.
    2. From all positive training bags.

    Evaluation Steps of MILIS:
    1. instance selection
    2. Feature mapping
    3. instance optimization - based on classification result.
    4. classification
    '''

    def __init__(self, glm, pol, sigma2):
        super(MILDM, self).__init__()
        self.glm = glm
        self.pol = pol
        self.sigma2 = sigma2
        self.m = 0
    
    # Evaluation.
    def evaluate(self, bags_features, bags_labels, fold,classifier):
        '''
        bags_features, bags_labels, fold, pol - positive bags only?, classifier
        '''
        self.N_fold_CV(bags_features, bags_labels, fold, self.glm, self.pol, self.sigma2, classifier)

    def N_fold_CV(self, all_bags, all_bags_labels, fold, global_mapping, postive_only, sigma2, classifier):
        '''
        all_bags, all_bags_labels, fold, global_mapping, postive_only, sigma2, classifier
        '''
        global_mapping = self.glm
        postive_only = self.pol
        sigma2 = self.sigma2

        result = []
        bag_indices = np.array(list(all_bags.keys()))
        kfold = KFold(fold, True, 1)
        
        train_bags = {}
        train_label = {}
        test_bags = {}
        test_label = {}
        fold_index = 1
        for train, test in kfold.split(bag_indices):
            #print('train: %s, test: %s' % (bag_indices[train], bag_indices[test]))
            
            #print("Fold {}:".format(fold_index))
            fold_index += 1
            # Build training bags
            for train_bag_index in bag_indices[train]:
                train_bags.update({train_bag_index:all_bags[train_bag_index]})
                train_label.update({train_bag_index:all_bags_labels[train_bag_index]})
            
            # Calculate L based on training bags
            L = self.cal_L(train_label)
            
            # Build test bags
            for test_bag_index in bag_indices[test]:
                test_bags.update({test_bag_index:all_bags[test_bag_index]})
                test_label.update({test_bag_index:all_bags_labels[test_bag_index]})
            
            # Perform test on each fold.
            result.append(self.One_fold_CV(train_bags, train_label, test_bags, test_label, global_mapping, postive_only, L, sigma2, classifier))

        auc = []
        f1 = []
        for rec in result:
            auc.append(rec["auc"])
            f1.append(rec["f1"])
        
        avg_auc = statistics.mean(auc)
        stdv_auc = statistics.stdev(auc)
        
        avg_f1 = statistics.mean(f1)
        stdv_f1 = statistics.stdev(f1)
        
        print("The MILDM's {} Fold test result on classifier {} and pol {}, gml {} is: ".format(fold, classifier, self.pol, self.glm))
        print("F1 score: {:.4f} +- {:.4f}, AUC: {:.4f} +- {:.4f} ".format(avg_f1, stdv_f1, avg_auc, stdv_auc))
        return result

    def cal_L(self, train_label):

        num = len(train_label.keys())
        num_positive_bag = 0
        num_negative_nag = 0
        for bag_index in train_label.keys():
            if train_label[bag_index] == 1:
                num_positive_bag += 1
            else:
                num_negative_nag += 1
        total = perm(num,2)
        B = perm(num_negative_nag,1) * perm(num_positive_bag,1)
        A = total - B
        
        Q = []
        for i in train_label.keys():
            i_label = train_label[i] 
            row = []
            for j in train_label.keys():
                j_label = train_label[j]
                ij_label = i_label * j_label
                row.append(-1/A if ij_label==1 else 1/B)
            Q.append(row)
        Q = np.array(Q)
        Q = sp.coo_matrix(Q)
        rowsum = np.array(Q.sum(1))
        D = sp.diags(rowsum.flatten())
        L = D-Q
        return L
    
    def One_fold_CV(self, train_bags, train_label, test_bags, test_label, global_mapping, postive_only, L, sigma2, classifier):
        
        '''
        train_bags, train_label, test_bags, test_label, global_mapping, postive_only, L, sigma2, classifier
        '''

        # Step1 Get the DIP.
        dip = self.DIP(train_bags, train_label, global_mapping, postive_only, L, sigma2)
        #print("The shape of DIP pool is:", dip[0].shape)
        
        # Step2 bag feature mapping. Training bags feature mapping and test bags feature mapping.
        train_bags_features = FeatureMapping(train_bags, dip, sigma2)
        test_bags_features = FeatureMapping(test_bags, dip, sigma2)
        
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
            # for rs in svm_result:
            #     print("SVM evaluation {} score is: {}".format(rs,svm_result[rs]))
            
            result = svm_result

        if classifier == "knn":
            # KNN classifier
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(arr_train_features, arr_train_label)
            #knn_score = knn.score(arr_test_features, arr_test_label)
            preds = knn.predict(arr_test_features)
            #print("SVM score:", clf.score(arr_test_features, arr_test_label))
            knn_result = evaluation(arr_test_label, preds)
            # for rs in knn_result:
            #     print("KNN evaluation {} score is: {}".format(rs,knn_result[rs]))
            
            result  = knn_result

        if classifier == "nb":
            # Naive Bayes - Gaussian naive bayes classifier
            gnb = GaussianNB()
            gnb.fit(arr_train_features, arr_train_label)
            preds = gnb.predict(arr_test_features)
            nb_result = evaluation(arr_test_label, preds)
            # for rs in nb_result:
            #     print("Gaussian Naive Bayes evaluation {} score is: {}".format(rs,nb_result[rs]))

            result = nb_result

        if classifier == "mlp":
            # MLP - DNN
            mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=100000)
            mlp.fit(arr_train_features, arr_train_label)           
            preds = mlp.predict(arr_test_features)
            mlp_result = evaluation(arr_test_label, preds)
            # for rs in mlp_result:
            #     print("MLP evaluation {} score is: {}".format(rs,mlp_result[rs]))
            result = mlp_result

        return result

    # Get the DIP instance pool
    def DIP(self, train_bags, train_label, global_mapping, postive_only, L, sigma2):
        
        P = []
        J = []
        tau = 0
        m_count = 0
        
        train_instances_set = []
        self.m = 0
        # a mapping or p mapping. - as inidcated in the paper.
        if postive_only: 
            #print("Building Instance Pool from positive bags only!")
            for bag_index in train_bags.keys():
                if train_label[bag_index] == 1:
                    for instance in train_bags[bag_index]:
                        train_instances_set.append(instance)
                    self.m += 1
        else:
            #print("Building Instance Pool from all bags!")
            for bag_index in train_bags.keys():
                for instance in train_bags[bag_index]:
                    train_instances_set.append(instance)
                self.m += 1
        
        train_instances_set = np.array(train_instances_set)
        #print(train_instances_set.shape)
        if global_mapping:
            # global instance selection - select the top m in all instances
            P = self.Global_selection(train_bags, train_instances_set, L, self.m, sigma2)
        else:
            # local instance selection - select one instance from each training bags.
            P = self.Local_selection(train_bags, train_instances_set, L, sigma2)

        return P

    def Global_selection(self, train_bags, train_instances_set, L, m, sigma2):
        
        P = []
        J = []
        tau = 0
        m_count = 0
        #print("Select Global instances to form the DIP!")
        for instance in train_instances_set:
            
            # calculate the similarity of each bag to this instance
            x_k = []
            for bag_index in train_bags.keys():
                sim = cal_each_feature(train_bags[bag_index], instance, sigma2)
                x_k.append(sim)
                
            x_k = sp.coo_matrix(x_k)
            
            # calculate f(x_k, L) based on Eq.(6) in the paper
            f = x_k.dot(L).dot(x_k.transpose())
            f = f.toarray()
            f_k = f[0][0]
            
            if m_count <= m or f_k > tau:
                P.append(instance)
                J.append(f_k)
                m_count += 1
                
            if m_count > m:
                tau_index = J.index(min(J))
                P.pop(tau_index)
                J.pop(tau_index)
                m_count -= 1

            tau = min(J)
        return P

    def Local_selection(self, train_bags, train_instances_set, L, sigma2):
        
        P = []
        J = []
        #print("Select local instances to form the DIP!")
        for bag_index in train_bags.keys():
            
            tau = 0
            # Select one instance inside each bag with the highest discriminative score.
            for instance in train_bags[bag_index]:
                
                # calculate the similarity of each bag to this instance
                x_k = []
                for bag_index in train_bags.keys():
                    sim = cal_each_feature(train_bags[bag_index], instance, sigma2)
                    x_k.append(sim)

                x_k = sp.coo_matrix(x_k)

                # calculate f(x_k, L) based on Eq.(6) in the paper
                f = x_k.dot(L).dot(x_k.transpose())
                f = f.toarray()
                f_k = f[0][0]
                
                if f_k > tau and len(P) == 0:
                    P.append(instance)
                    J.append(f_k)
                    tau = f_k
                    
                elif f_k > tau and len(P) > 0:
                    P.pop()
                    J.pop()
                    P.append(instance)
                    J.append(f_k)
                    tau = f_k
            
        return P