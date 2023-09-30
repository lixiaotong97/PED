#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import numpy as np

import json
import time

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score, PED_Score

models_hub = ['byol', 'deepcluster-v2', 'infomin', 'insdis', 'moco-v1',
              'moco-v2', 'pcl-v1', 'pcl-v2', 'sela-v2', 'simclr-v1',
              'simclr-v2', 'swav']



def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)

def save_time(time_dict, fpath):
    with open(fpath, "w") as f:
        # write dict
        json.dump(time_dict, f)

def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False

def exist_time(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via logistic regression.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='logme',
                        help='name of the method for measuring transferability')
    parser.add_argument('--nleep-ratio', type=float, default=5,
                        help='the ratio of the Gaussian components and target data classess')
    parser.add_argument('--parc-ratio', type=float, default=2,
                        help='PCA reduction dimension')
    parser.add_argument('--output-dir', type=str, default='./results_metrics_cr/',
                        help='dir of output score')
    parser.add_argument('--time_dir', type=str, default='./results_metrics_cr/time/Ours_time')
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--num_step', type=int, default=5)
    parser.add_argument('--rescale', action='store_true', default=True)
    parser.add_argument('--nsigma', type=float, default=0.6) # We have two candidates for the hyperparameter nsigma 0.3 0.6
    parser.add_argument('--type', type=str, default='SFDA')
    parser.add_argument('--er', type=float, default=0.5)
    parser.add_argument('--constant', type=float, default=1)

    args = parser.parse_args()
    pprint(args)

    score_dict = {}
    metric = args.metric

    fpath = os.path.join(args.output_dir, metric)
    # else:
    #     fpath = os.path.join(args.output_dir, args.metric)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.dataset}_metrics.json')

    if not os.path.exists(fpath):
        save_score(score_dict, fpath)
    else:
        with open(fpath, "r") as f:
            score_dict = json.load(f)

    for model in models_hub:
        args.model = model

        model_npy_feature = os.path.join('./results_f/', f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join('./results_f/', f'{args.model}_{args.dataset}_label.npy')
        X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)

        print(f'x_trainval shape:{X_features.shape} and y_trainval shape:{y_labels.shape}')
        print(f'Calc Transferabilities of {args.model} on {args.dataset}')

        if args.metric == 'logme':
            score_dict[args.model] = LogME_Score(X_features, y_labels)
        elif args.metric == 'nleep':
            ratio = 1 if args.dataset in ('food', 'pets') else args.nleep_ratio
            score_dict[args.model] = NLEEP(X_features, y_labels, component_ratio=ratio)
        elif args.metric == 'sfda':
            score_dict[args.model] = SFDA_Score(X_features, y_labels)
        elif args.metric == 'PARC':
            score_dict[args.model] = PARC_Score(X_features, y_labels)
        elif args.metric == 'PED':
            feature_path = "./results_f"
            general_name = f'{args.model}_imagenet' # use val set as default
            general_model_npy_feature = os.path.join(feature_path, general_name + '_feature.npy')
            general_model_npy_label = os.path.join(feature_path, general_name + '_label.npy')
            general_X_features, general_y_labels = np.load(general_model_npy_feature), np.load(general_model_npy_label)
            ratio = 1 if args.dataset in ('food', 'pets') else args.nleep_ratio
            score_dict[args.model] = PED_Score(X_features, y_labels, general_X=general_X_features, exit_ratio=args.er, time_step=args.step, type=args.type,
                                                   component_ratio=ratio, nsigma=args.nsigma, constant=args.constant)
        else:
            raise NotImplementedError

        print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
        save_score(score_dict, fpath)

    results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
    print(f'Models ranking on {args.dataset} based on {args.metric}: ')
    pprint(results)
    results = {a[0]: a[1] for a in results}
    save_score(results, fpath)
