# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/08/29 12:15:27
@Author  :   Xiaoxiao Ma, Jia Wu
@Version :   1.0
@Contact :   xiaoxiao.ma2@hdr.mq.edu.au
@Desc    :   Contact author for details
'''

# here put the import lib
import argparse
import os
import sys
from utils import load_data
from MILES import MILES
from MILIS import MILIS
from MILDM import MILDM

def parse_args():
    '''
    Parses the MIL model arguments.
    '''
    parser = argparse.ArgumentParser("MIL",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')

    parser.add_argument('--dataset', nargs='?', default='Musk1',
                        help='name of dataset. Available Datasets: Musk1, Musk2, Elephant, Tiger, Ng9, Ng18, web7,web8')	

    parser.add_argument('--glm', default=True, type=bool,
                        help='global or local instance selection. Boolean.')
                        
    parser.add_argument('--pol', default=False, type=bool,
                        help='select instances from all packages or from positive packages. Boolean.')

    parser.add_argument('--classifier', nargs='?', default='knn',
                        help='classifier: 1.svm, 2.knn, 3.naive bayes, 4.mlp. Default is svm')
    
    parser.add_argument('--model', nargs='?', default='MILES',
                        help='models: MILES, MILIS, MILDM')

    parser.add_argument('--sigma2', default=800000, type=int,
                        help='sigma squared used to calculate the similarity between bags and instances')

    parser.add_argument('--fold', default=10, type=int,
                        help='Fold of cross validation. Default is 10-Fold Cross validation')

    return parser.parse_args()

def main(args):
    
    dataset = args.dataset
    glm = args.glm
    pol = args.pol
    classifier = args.classifier
    sigma2 = args.sigma2
    fold = args.fold

    model_option = args.model

    # Read data from datasets
    bags_features, bags_labels = load_data(dataset)

    # Evaluate the model.
    if model_option == "MILES":
        #print("Evaluating MILES!")
        model = MILES(sigma2)
        model.evaluate(bags_features, bags_labels, fold, classifier)

    if model_option == "MILIS":
        beta = 1/sigma2
        k = 10
        z = 1
        #print("Evaluating MILIS!")
        model = MILIS(beta, k, z)
        model.evaluate(bags_features, bags_labels, fold, pol, classifier)

    if model_option == "MILDM":
        #print("Evaluating MILDM!")
        model = MILDM(glm, pol, sigma2)
        model.evaluate(bags_features, bags_labels, fold, classifier)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    