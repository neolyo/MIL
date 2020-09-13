# -*- encoding: utf-8 -*-
'''
@File    :   MILIS.py
@Time    :   2020/08/29 12:28:44
@Author  :   Xiaoxiao Ma, Jia Wu
@Version :   1.0
@Contact :   xiaoxiao.ma2@hdr.mq.edu.au
@Desc    :   Contact author for details
@Original Paper: MILIS: Multiple Instance Learning with Instance Selection
@Paper Link:     https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5557878
'''

import numpy as np
from scipy.spatial import distance

from scipy.optimize import linprog
import math
import statistics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from utils import load_data, FeatureMapping, evaluation, K_neg_instances, KDE

class MILIS():
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

    def __init__(self, beta, k, z):
        super(MILIS, self).__init__()
        self.beta = beta
        self.k = k
        self.z = z
    
    # Evaluation.
    def evaluate(self, bags_features, bags_labels, fold, pol, classifier):
        '''
        bags_features, bags_labels, fold, pol - positive bags only?, classifier
        '''
        self.N_fold_CV(bags_features, bags_labels, fold, pol, classifier)

    def N_fold_CV(self, all_bags, all_bags_labels, fold, pol, classifier):
        
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
            result.append(self.One_fold_CV(train_bags, train_label, test_bags, test_label, pol, classifier))

        auc = []
        f1 = []
        for rec in result:
            auc.append(rec["auc"])
            f1.append(rec["f1"])
        
        avg_auc = statistics.mean(auc)
        stdv_auc = statistics.stdev(auc)
        
        avg_f1 = statistics.mean(f1)
        stdv_f1 = statistics.stdev(f1)
        
        print("The MILIS's {} Fold test result with classifier {} and pol {} is: ".format(fold, classifier, pol))
        print("F1 score: {:.4f} +/- {:.4f}, AUC: {:.4f} +/- {:.4f} ".format(avg_f1, stdv_f1, avg_auc, stdv_auc))

    def One_fold_CV(self, train_bags, train_label, test_bags, test_label, pol, classifier):
        
        # For MILIS, the instance pool is constructed by representative instances from each bags. One representative Instance from each bag.
        # Steps:
        # 1. Initial instance selection
        # 2. Train classifer
        # 3. Instance update based on the classification result
        # 4. After iterating N times, An optimal instance pool is selected.
        # 5. Test feature mapping and classification result evaluation.

        # Step1 Initial instance selection
        IIP, phi = self.init_instance_selection(train_bags, train_label, pol)

        # Step2 Train SVM and perform instance update
        sigma2 = 1/self.beta
        #IIP = self.instance_update(train_bags, IIP, train_label, phi, sigma2)

        train_bags_features = FeatureMapping(train_bags, IIP, sigma2)
        test_bags_features = FeatureMapping(test_bags, IIP, sigma2)

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
        
        # Train the final classifier and evaluate
        if classifier == "svm":
            # SVM classifier
            clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, max_iter=10000)
            clf.fit(arr_train_features, arr_train_label)
            preds = clf.predict(arr_test_features)
            svm_result = evaluation(arr_test_label, preds)
            #for rs in svm_result:
            #    print("SVM evaluation {} score is: {}".format(rs,svm_result[rs]))
            
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

    def init_instance_selection(self, bags, bags_label, positive_only):
        
        '''
        In all bags case:
        For each bag, select a most discriminative instance to form the IIP.
            In positive bags, select the least negative one, using knn.
            In negative bags, select the most negative one, using knn
        
        In positive bags case:
        For each positive bag, select a most discriminative instance to form the IIP.
            The positive only case can be regared as a special case of the all bags case.
            
        Select the least negative one in postive bag - the lowest PDF
        Select the most negative one in negative bag - the biggest PDF
        '''
        
        # Build the negative instance pool to estimate the negative PDF
        all_neg_instances = []
        for bag in bags_label.keys():
            if bags_label[bag] == -1:
                for instance in bags[bag]:
                    all_neg_instances.append(instance)
        all_neg_instances = np.array(all_neg_instances)
        
        # Train a KNN based on all negative instances.
        knn = NearestNeighbors(n_neighbors = self.k)
        knn.fit(all_neg_instances)
        
        # Select instances for the mapping pool. If positive_only, select from positive bags, else all bags   
        proto_bags = {}
        if positive_only:
            for bag in bags_label.keys():
                if bags_label[bag] == 1:
                    proto_bags.update({bag:bags[bag]})
        else:
            proto_bags = bags
        
        # For each instance in the prototype bags, select one instance in the bag using knn and its PDF.
        IIP = []
        phi = {} # store the selected instance in each bag.
        for bag in proto_bags.keys():
            
            least_negative_p = 100000
            most_negative_p = 0
            for instance in proto_bags[bag]:
                kn_negs_index = K_neg_instances(instance, knn)
                kn_negs = []
                for i in kn_negs_index:
                    kn_negs.append(all_neg_instances[i])
                
                # Positive bags
                if bags_label[bag] == 1:
                    cur_kde = KDE(instance, kn_negs, self.z, self.beta)
                    if cur_kde < least_negative_p:
                        least_negative_p = cur_kde
                        least_negative_instance = instance
                        
                # Negative bags
                if bags_label[bag] == -1:
                    cur_kde = KDE(instance, kn_negs, self.z, self.beta)
                    if cur_kde > most_negative_p:
                        most_negative_p = cur_kde
                        most_negative_instance = instance    
            
            if bags_label[bag] == 1:
                IIP.append(least_negative_instance)
                phi.update({bag: least_negative_instance})
            else:
                IIP.append(most_negative_instance)
                phi.update({bag: most_negative_instance})
            
        return IIP, phi

    def instance_update(self, train_bags, instance_pool, train_label, phi, sigma2):
        
        '''
        train_bags, train_bags_features, train_label, phi, sigma2
        
        bags_feature is a dict of : { bag_index : bag_mapped_feature}
        iteration: Instance updating iteration
        init_svm_weights: weights of the trained svm in the last iteration.
        '''

        ### Add instance update features here.
        IIP = instance_pool
        for iteration in range(5):
            
            arr_train_features = []
            arr_train_label = []

            train_bags_features = FeatureMapping(train_bags, IIP, sigma2)

            for bag in train_bags_features.keys():
                arr_train_features.append(train_bags_features[bag])
                arr_train_label.append(train_label[bag])

            arr_train_features = np.array(arr_train_features)
            arr_train_label = np.array(arr_train_label)
            
            clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True,max_iter=10000)
            clf.fit(arr_train_features, arr_train_label)
            params = clf.coef_

            bags_loss = {}
            phi = phi
            v = 0
            for bag_index in train_bags.keys():
                prediction = params @ train_bags_features[bag_index]
                #loss = max(0, 1 - bags_label[bag]*prediction) ** 2
                loss = self.cal_loss(train_label[bag_index], prediction)
                bags_loss.update({bag_index:loss})
                v += loss
        
            # Only update IIP that contributes to the phi.
            j = 0
            for bag_index in phi.keys():
                if bags_loss[bag_index] > 0:
                    for instance in train_bags[bag_index]:
                        if not np.array_equal(instance, phi[bag_index]):
                            v_prim = self.feature_update(train_bags, train_bags_features, train_label, phi, instance, j, prediction, bags_loss, params, v, sigma2)
                            if v_prim < v:
                                #print("updated an instance!")
                                phi.update({bag_index: instance})
                                j += 1
                                break
                j += 1
        
            IIP = []
            for bag_index in phi.keys():
                IIP.append(phi[bag_index])
                        
        return IIP

    def feature_update(self, bags, bags_feature, bags_label, phi, new_phi_j, j, prediction, bags_loss, svm_weights, v, sigma2):
        
        v_prim = 0
        for bag_index in bags.keys():
            pi_j = bags_feature[bag_index][j]
            pi_j_prim = self.cal_each_feature(bags[bag_index], new_phi_j, sigma2)
            prediction_prim = prediction + svm_weights[0][j]*(pi_j_prim - pi_j)
            
            v_prim += self.cal_loss(bags_label[bag_index], prediction_prim)
            
            if v_prim >= v:
                return v_prim
        
        return v_prim

    def cal_loss(self, label, prediction):
        return max(0, 1 - label*prediction) ** 2

    def cal_each_feature(self, bag, instance_ip, sigma2):
    
        max_sim = 0
        for instance in bag:
            dist = distance.euclidean(instance, instance_ip)
            k = math.exp(-(math.sqrt(dist))/(sigma2))
            if k > max_sim:
                max_sim = k
        
        return max_sim