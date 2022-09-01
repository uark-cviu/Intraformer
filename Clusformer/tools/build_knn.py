import os
import math
import multiprocessing as mp
import numpy as np 
import argparse
from evaluation import *
from tqdm import *
from tools.knn import build_knns
from utils.misc import read_meta, l2norm, read_probs


def get_args_parser():
    parser = argparse.ArgumentParser('Build K-NN', add_help=False)
    # dataset parameters
    parser.add_argument('--feature_path', 
                        default='./data/labels/part1_test.meta')
    parser.add_argument('--label_path', default='./data/part1_test/faiss_k_64.npz')
    parser.add_argument('--knn_method', default='faiss')
    parser.add_argument('--out_dir', default='./data/our_knns/')
    parser.add_argument('--k', default=64, type=int)
    parser.add_argument('--feature_dim', default=64, type=int)
    parser.add_argument('--is_rebuild', default=True, type=bool)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Build K-NN script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)

    lb2idxs, idx2lb = read_meta(args.label_path)
    inst_num = len(idx2lb)
    features = read_probs(args.feature_path, inst_num, args.feature_dim)
    features = l2norm(features)
    print("Feature shape ", features.shape)
    # features = features[:100, :]
    knns = build_knns(args.out_dir,
                    features,
                    args.knn_method,
                    args.k,
                    num_process=4,
                    is_rebuild=args.is_rebuild,
                    feat_create_time=None)