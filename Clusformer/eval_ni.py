import numpy as np 
import argparse
from evaluation import *
from tqdm import *
from utils.misc import read_meta, intdict2ndarray, Timer
from scipy.sparse import csr_matrix


def get_args_parser():
    parser = argparse.ArgumentParser('Set clustering transformer', add_help=False)
    # dataset parameters
    parser.add_argument('--label_path', 
                        default='./data/labels/part1_test.meta')
    parser.add_argument('--knn_graph_path', 
                        default='./data/part1_test/faiss_k_64.npz')
    parser.add_argument('--predict_path', 
                        default='/home/ubuntu/checkpoints/reproduce/valid_prediction.npy')
    parser.add_argument('--threshold', default=0.9, type=float)
    parser.add_argument('--tau_0', default=0.65, type=float)
    return parser


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build_connections(nbrs, pred, th, **kwargs):
    n_cluster = len(pred)

    edges = []
    for i in tqdm(range(n_cluster), total=n_cluster):
        pred_ = pred[i]
        idx = np.where(pred_ > th)[0]
        nbrs_ = nbrs[i, idx][1:]
        cluster_id = [i] * len(nbrs_)
        for a, b in zip(cluster_id, nbrs_):
            edges.append([a, b])

    values=[1]*len(edges)
    edges=np.array(edges)

    return values, edges 


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while (u != parent[u]):
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    
    with Timer("Load prediction: "):
        pred = np.load(f"{args.predict_path}/valid_predict_1.npy")
    pred = sigmoid(pred) 
    # import pdb; pdb.set_trace()

    lb2idxs, idx2lb = read_meta(args.label_path)
    inst_num = len(idx2lb)
    gt_labels = intdict2ndarray(idx2lb)
    with Timer("Load KNN graph: "):
        knns = np.load(args.knn_graph_path)['data']

    nbrs = knns[:, 0, :].astype(np.int)
    dists = knns[:, 1, :]

    values, edges = build_connections(nbrs, pred, args.threshold)

    adj2 = csr_matrix((values, (edges[:,0].tolist(), edges[:,1].tolist())), shape=(inst_num, inst_num))
    link_num = np.array(adj2.sum(axis=1))
    common_link = adj2.dot(adj2)

    # threshold2 = 0.65
    with Timer('Second step'):
        edges_new = []
        edges = np.array(edges)
        share_num = common_link[edges[:,0].tolist(), edges[:,1].tolist()].tolist()[0]
        edges = edges.tolist()

        for i in range(len(edges)):
            if ((link_num[edges[i][0]]) != 0) & ((link_num[edges[i][1]]) != 0):
                if max((share_num[i])/link_num[edges[i][0]],(share_num[i])/link_num[edges[i][1]])>args.tau_0:
                    edges_new.append(edges[i])
            if i%10000000==0:
                print(i)

    with Timer('Last step'):
        pre_labels = edge_to_connected_graph(edges_new, inst_num)
        evaluate(gt_labels, pre_labels, 'pairwise')
        evaluate(gt_labels, pre_labels, 'bcubed')
        evaluate(gt_labels, pre_labels, 'nmi')