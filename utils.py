import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.utils import (negative_sampling, add_self_loops,train_test_split_edges,to_undirected)
import scipy.io as scio


def random_edge_mask(args, split_edge, device, num_nodes):
    edge_index = split_edge['train']['edge']
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * args.mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].t()

    edge_index = to_undirected(edge_index_train)
    
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device) 


def train_test_split_edges_direct(data, val_ratio: float = 0.05,
                           test_ratio: float = 0.1):
    

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    #edge_attr = data.edge_attr
    edge_attr = None
    data.edge_index = data.edge_attr = None

    
    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    #perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    perm = torch.randperm(neg_row.size(0))[:(row.size(0)*1)]          
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    
    n_v = n_v*1
    n_t = n_t*1
    
    r, c = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([r, c], dim=0)
    
    r, c = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([r, c], dim=0)
    
    r, c = neg_row[n_v + n_t:], neg_col[n_v + n_t:]
    data.train_neg_edge_index = torch.stack([r, c], dim=0)

    return data

def edge_split_direct(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1): #val_ratio=0.05, test_ratio=0.1
    data = dataset
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges_direct(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        #print("row.size(0): ",row.size(0))
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    """
    print("split_edge['train']['edge']: ",split_edge['train']['edge'].shape)
    print("split_edge['train']['edge_neg']: ",split_edge['train']['edge_neg'].shape)
    print("split_edge['valid']['edge']: ",split_edge['valid']['edge'].shape)
    print("split_edge['valid']['edge_neg']: ",split_edge['valid']['edge_neg'].shape)
    print("split_edge['test']['edge']: ",split_edge['test']['edge'].shape)
    print("split_edge['test']['edge_neg']: ",split_edge['test']['edge_neg'].shape)
    """
    return split_edge


def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true):
    
    train_auc = roc_auc_score(train_true, train_pred)
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    train_ap = average_precision_score(train_true, train_pred)
    valid_ap = average_precision_score(val_true, val_pred)
    test_ap = average_precision_score(test_true, test_pred)
    
    
    pred = test_pred.clone().detach()
    labels = test_true.clone().detach()
    threshold = 0.55
    for i in range(train_pred.shape[0]):
        if train_pred[i] >= threshold:
            train_pred[i] = 1
        else:
            train_pred[i] = 0
    
     
    for i in range(val_pred.shape[0]):
        if val_pred[i] >= threshold:
            val_pred[i] = 1
        else:
            val_pred[i] = 0
            
    
    for i in range(test_pred.shape[0]):
        if test_pred[i] >= threshold:
            test_pred[i] = 1
        else:
            test_pred[i] = 0
            
    train_aps = average_precision_score(train_true, train_pred)
    valid_aps = average_precision_score(val_true, val_pred)
    test_aps = average_precision_score(test_true, test_pred)
    
    
    
    f1_test = f1_score(test_true,test_pred)
    
    results = dict()
    results['AUC'] = (train_auc, valid_auc, test_auc,f1_test)
    results['AUPR'] = (train_ap, valid_ap, test_ap,f1_test) 
    
    return results


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self,key, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            
            
            result =  torch.tensor(self.results)

            best_results = []
            for r in result:
                
                train1 = 100 * r[:, 0].max().item()
                valid = 100 * r[:, 1].max().item()
                train2 = 100 * r[r[:, 1].argmax(), 0].item()
                test = 100 * r[r[:, 3].argmax(), 2].item()
                f1 = r[r[:, 3].argmax(), 3].item()
                
                best_results.append((train1, valid, train2, test, f1))

            best_result = torch.tensor(best_results)

            r = best_result[:, 3]
            print(f'   Final {key}: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final F1: {r.mean():.2f} ± {r.std():.2f}')
            









