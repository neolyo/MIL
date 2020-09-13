# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/09/07 15:16:47
@Author  :   Xiaoxiao Ma, Jia Wu
@Version :   1.0
@Contact :   xiaoxiao.ma2@hdr.mq.edu.au
@Desc    :   Contact author for details
'''

# here put the import lib

import numpy as np
import math
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def load_data(dataset):
    '''
    Load data from records
    '''

    # dataset stored in txt format
    if dataset == "Musk1":
        filepath = 'datasets/MUSK1/clean1/clean1.data'
        bags_features, bags_labels = load_txt_data(filepath)
    if dataset == "Musk2":
        filepath = 'datasets/MUSK1/clean2/clean2.data'
        bags_features, bags_labels = load_txt_data(filepath)
    
    # dataset stored in csv format
    if dataset == "Elephant":
        filepath = "datasets/{}.csv".format(dataset)
        bags_features, bags_labels = load_csv_data(filepath)
    if dataset == "Tiger":
        filepath = "datasets/{}.csv".format(dataset)
        bags_features, bags_labels = load_csv_data(filepath)
    if dataset == "Ng9":
        filepath = "datasets/{}.csv".format(dataset)
        bags_features, bags_labels = load_csv_data(filepath)    
    if dataset == "Ng18":
        filepath = "datasets/{}.csv".format(dataset)
        bags_features, bags_labels = load_csv_data(filepath)
    if dataset == "Web7":
        filepath = "datasets/{}.csv".format(dataset)
        bags_features, bags_labels = load_csv_data(filepath)
    if dataset == "Web8":
        filepath = "datasets/{}.csv".format(dataset)
        bags_features, bags_labels = load_csv_data(filepath)
    
    return bags_features, bags_labels

def load_csv_data(filepath):
    
    record = pd.read_csv(filepath, header = None) 
    
    bags_labels = {}
    bags_names = {}
    bags_features = {}
    bag_id = 0
    instances_features = []

    for index, rec in record.iterrows():
        rec_list = list(rec)
        bag_label = rec_list[0]
        bag_name = rec_list[1]

        if bag_name not in bags_names:
            bags_names.update({bag_name:bag_id})

            if int(float(bag_label)) == 1:
                bags_labels.update({bags_names[bag_name]:1})
            else:
                bags_labels.update({bags_names[bag_name]:-1})
            bag_id += 1

    bags_features = { id : [] for id in range(bag_id)}
    
    num_instances = 0
    
    for index, rec in record.iterrows():

        rec_list = list(rec)
        bag_name = rec_list[1]
        features = rec_list[2:]
        
        bags_features[bags_names[bag_name]].append(features)
        num_instances += 1
    print("number of instances in this dataset is : ", num_instances)

    for bag_index in bags_features.keys():
        instances = bags_features[bag_index]
        instances_features = np.array(instances, dtype=np.float32)
        bags_features.update({bag_index : instances_features})

    return bags_features, bags_labels

def load_txt_data(filepath):
    bags_labels = {}
    bags_names = {}
    bags_features = {}
    bag_id = 0
    instances_features = []
    with open(filepath) as fp:

        for line in fp:

            bag_name = line.strip().split(',')[0]
            features = line.strip().split(',')[2:-1]
            bag_label = line.strip().split(',')[-1]

            if bag_name not in bags_names:
                bags_names.update({bag_name:bag_id})
                
                if int(float(bag_label)) == 1:
                    bags_labels.update({bags_names[bag_name]:1})
                else:
                    bags_labels.update({bags_names[bag_name]:-1})
                instances = []
                bag_id += 1

            instances.append(features)
            instances_features = np.array(instances,dtype=np.int32)
            bags_features.update({bags_names[bag_name] : instances_features})

    return bags_features, bags_labels

def FeatureMapping(bags, ip, sigma2):
    
    '''
    Map each bag in bags to the similarity space with IP
    bags is a dict of bag, with bag index as key and instances features as content
    ip is the selected instance prototypes, a matrix of instance features
    sigma is the parameter used to calculate the similarity
    return a dict of bags, bag index as the key, mapped features as content
    '''
    
    bags_features = {}
    
    for bag_index in bags.keys():
        
        bag_mapped_feature = []
        for instance in ip:
            max_sim = 0
            for instance_in_bag in bags[bag_index]:
                dist = distance.euclidean(instance_in_bag, instance)
                k = math.exp(-1 * (math.sqrt(dist))/(sigma2))
                if k > max_sim:
                    max_sim = k

            bag_mapped_feature.append(max_sim)
        feature = np.array(bag_mapped_feature)
        bags_features.update({bag_index:feature})
        
    return bags_features

def cal_each_feature(bag, instance_ip, sigma2):
    '''
    Calculate a bag's feature on a single instance - that is S(B_i, IP_j).
    '''
    max_sim = 0
    for instance in bag:
        dist = distance.euclidean(instance, instance_ip)
        k = math.exp(-(math.sqrt(dist))/(sigma2))
        if k > max_sim:
            max_sim = k
    
    return max_sim

def evaluation(y_true, y_pred):
    
    result = {}
    
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    result.update({"f1" : f1})
    result.update({"auc" : roc_auc})
    
    return result

def K_neg_instances(x, knn):
    '''
    Get the k nearest negative instances from the negative instance pool.
    Returns the indices of k nearest instances.
    '''
    x = np.reshape(x,(1,x.shape[0]))
    res = knn.kneighbors(x, return_distance=False)
    k_res = []
    for i in res[0]:
        k_res.append(i)
    
    # return the indices of k nearest neighbors.
    return k_res

def KDE(x, kn_negs, Z, beta):
    '''
    Calculate the density estimation using Gaussian-kernal-based kernel density estimator
    '''
    num_neg_instances = 0
    p = 0
    x = np.reshape(x,(1,x.shape[0]))
    
    for instance in kn_negs:
        num_neg_instances += 1
        inst = np.reshape(instance,(1,instance.shape[0]))
        dist = manhattan_distances(x,inst)
        p = p + math.exp(-1 * beta * dist)
        
    p = p / (Z * num_neg_instances)
    return p
