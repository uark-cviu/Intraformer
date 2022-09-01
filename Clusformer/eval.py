import numpy as np 
import argparse
from evaluation import *
from tqdm import *
from utils.misc import read_meta, intdict2ndarray, Timer


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


def s_nbr(dists, nbrs, pred, th, **kwargs):
    ''' use supervised confidence defined on neigborhood
    '''
    num, _ = dists.shape
    conf = np.zeros((num, ), dtype=np.float32)
    contain_neg = 0
    for i, (nbr, dist) in tqdm(enumerate(zip(nbrs, dists)), total=len(nbrs)):
        # lb = idx2lb[i]
        pos, neg = 0, 0
        for j, n in enumerate(nbr):
            if pred[i][j] > th:
                pos += 1 - dist[j]
            else:
                neg += 1 - dist[j]
        conf[i] = pos - neg
        if neg > 0:
            contain_neg += 1
    print('#contain_neg:', contain_neg)
    conf /= np.abs(conf).max()
    return conf


def confidence_to_peaks(dists, nbrs, confidence, max_conn=1):
    # Note that dists has been sorted in ascending order
    assert dists.shape[0] == confidence.shape[0]
    assert dists.shape == nbrs.shape

    num, _ = dists.shape
    dist2peak = {i: [] for i in range(num)}
    peaks = {i: [] for i in range(num)}

    for i, nbr in tqdm(enumerate(nbrs)):
        nbr_conf = confidence[nbr]
        for j, c in enumerate(nbr_conf):
            nbr_idx = nbr[j]
            if i == nbr_idx or c <= confidence[i]:
                continue
            dist2peak[i].append(dists[i, j])
            peaks[i].append(nbr_idx)
            if len(dist2peak[i]) >= max_conn:
                break
    return dist2peak, peaks


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


def peaks_to_edges(peaks, dist2peak, tau):
    edges = []
    for src in peaks:
        dsts = peaks[src]
        dists = dist2peak[src]
        for dst, dist in zip(dsts, dists):
            if src == dst or dist >= 1 - tau:
                continue
            edges.append([src, dst])
    return edges


def peaks_to_labels(peaks, dist2peak, tau, inst_num):
    edges = peaks_to_edges(peaks, dist2peak, tau)
    pred_labels = edge_to_connected_graph(edges, inst_num)
    return pred_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    
    with Timer("Load prediction: "):
        pred = np.load(f"{args.predict_path}/valid_predict.npy")
    pred = sigmoid(pred)
    # import pdb; pdb.set_trace()

    lb2idxs, idx2lb = read_meta(args.label_path)
    inst_num = len(idx2lb)
    gt_labels = intdict2ndarray(idx2lb)
    with Timer("Load KNN graph: "):
        knns = np.load(args.knn_graph_path)['data']

    nbrs = knns[:, 0, :].astype(np.int)
    dists = knns[:, 1, :]
    
    with Timer("Compute confidence: "):
        conf = s_nbr(dists, nbrs, pred, args.threshold)
    pred_dist2peak, pred_peaks = confidence_to_peaks(dists, nbrs, conf, max_conn=1)

    
    best_pred_labels = None
    best_pairwise = -1
    best_bcubed = -1
    best_nmi = -1
    best_th = -1
    for tau_0 in np.arange(0.3, 1.0, 0.01):
        print(tau_0)
        pred_labels = peaks_to_labels(pred_peaks, pred_dist2peak, tau_0, inst_num)
        # Save predict labels 
        

        results = []
        for metric in ['pairwise', 'bcubed', 'nmi']:
            result = evaluate(gt_labels, pred_labels, metric)
            results.append(result)

        if results[0] > best_pairwise:
            best_pairwise = results[0]
            best_pred_labels = pred_labels
            best_bcubed = results[1]
            best_nmi = results[2]
            best_th = tau_0

        np.save(f"{args.predict_path}/pred_labels_th{args.threshold:.2f}_tau{tau_0:.2f}.npy", pred_labels)

    np.save(f"{args.predict_path}/pred_labels_best.npy", best_pred_labels)
    print(f"{best_pairwise} {best_bcubed} {best_nmi} {best_th}")
    with open(f"{args.predict_path}/result.txt", 'w') as f:
        f.write(f"{best_pairwise} {best_bcubed} {best_nmi} {best_th}")