import os
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs


def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


class ClusterDataset(Dataset):
    def __init__(self, feat_path, label_path, knn_graph_path, feat_dim, train=True):
        super(ClusterDataset, self).__init__()

        self.lb2idxs, self.idx2lb = read_meta(label_path)
        self.inst_num = len(self.idx2lb)
        self.gt_labels = intdict2ndarray(self.idx2lb)

        self.features = read_probs(feat_path, self.inst_num, feat_dim)

        self.features = l2norm(self.features)
        self.knns = np.load(knn_graph_path, allow_pickle=True)['data']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        knn = self.knns[index]
        nbrs = knn[0, :].astype(np.int)
        dists = knn[1, :]

        feature = self.features[nbrs] # K x D

        similarities = cosine_similarity(feature) # K x K
        feature = np.concatenate([feature, similarities], axis=1)

        targets = [1]
        prior_label = self.idx2lb[nbrs[0]]
        for i in range(1, len(nbrs)):
            if self.idx2lb[nbrs[i]] == prior_label:
                targets.append(1)
            else:
                targets.append(0)

        targets = np.asarray(targets).astype(np.float32)


        return {
            "features": feature,
            "targets": targets
        }