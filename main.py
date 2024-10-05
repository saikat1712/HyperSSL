import os
import os.path as osp
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_sparse import SparseTensor
from model import GCN_mgae, LPD 
from hyperEmbedding import HyperEmbedding
from utils import edge_split_direct, random_edge_mask, Logger, evaluate_auc
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling




    

def train(model, predictor, data, split_edge, optimizer, args):
    model.train()
    predictor.train()

    new_edge_index=[]

    adj, edge_index, edge_index_mask = random_edge_mask(args, split_edge, data.x.device, data.num_nodes)
    data.edge_index = adj.to(data.x.device)
    pos_train_edge = edge_index_mask
    new_edge_index = edge_index.cpu()   
    
    
    optimizer.zero_grad()
    
    h = model(data.x, adj)
    edge = pos_train_edge.T 
    pos_out = predictor(h, edge)
    pos_loss = -torch.log(pos_out + 1e-15).mean()

    # pos_edge = split_edge['train']['edge'].t()
    #new_edge_index, _ = add_self_loops(edge_index.cpu())
    
    
    edge = split_edge['train']['edge_neg']
    edge = edge.to(data.x.device)
    edge = edge.T  
    

    neg_out = predictor(h, edge)
    #print("neg_out:", neg_out)
    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

    loss = pos_loss + neg_loss
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, predictor, data, adj, split_edge, batch_size):
    model.eval()
    h = model(data.x, adj)
    #print("h: ",h)
    #print("h[0].shape: ",len(h))
    #h = h[0]
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        #print("edge_test: ",edge)
        pos_train_preds += [predictor(h, edge).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        #print("edge_test: ",edge)
        pos_valid_preds += [predictor(h, edge).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        #print("edge_test: ",edge)
        neg_train_preds += [predictor(h, edge).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        #print("type(pos_test_edge): ",type(pos_test_edge))
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    
    train_pred = torch.cat([pos_train_pred, neg_train_pred], dim=0)
    train_true = torch.cat([torch.ones_like(pos_train_pred), torch.zeros_like(neg_train_pred)], dim=0)

    val_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    val_true = torch.cat([torch.ones_like(pos_valid_pred), torch.zeros_like(neg_valid_pred)], dim=0)
    
    

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
    
    results = evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true)
    return results
    

def getPreEmbedding(data,prembsize):
    no_gene = len(data['gene']['node_id'])
    no_aux = len(data['aux']['node_id'])
    X = torch.rand(no_gene+no_aux,prembsize) #gene and aux node embeddings are initialized with random values, to be learnt
    lbl_gene = torch.zeros(no_gene) #gene nodes are assigned label 0
    lbl_aux = torch.ones(no_aux) #auxiliary nodes are assigned label 1
    lbl = torch.cat((lbl_gene,lbl_aux))
    lbl = lbl.type(torch.LongTensor)
    Bipartite = data['associated_to']['edge_index']
    HyperEmbedding(X,lbl,no_gene,Bipartite)
    embeddings = torch.load('prembed.pt')
    return embeddings     




def HyperSSL(data_name,args):
    

    
    prembsize = 128
    
    #print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    device = torch.device('cpu')


    dataset = torch.load(f"datasets/{data_name}.pt") 
    #print(dataset)
    
    #Stage1. HyperEmbedding Learning
    embeddings = getPreEmbedding(dataset,prembsize)
    #print("embeddings: ",embeddings)
    
    #Stage2. Masked AutoEncoder Link Prediction
    geneCoexpression = Data()
    geneCoexpression.num_nodes = len(dataset['gene']['node_id'])
    geneCoexpression.edge_index = dataset['interacts']['edge_index']
    geneCoexpression.x = embeddings
    geneCoexpression.num_features = 128
    all_edges = geneCoexpression.edge_index
    
    
    dataset.num_nodes = len(dataset['gene']['node_id'])
    dataset.edge_index = dataset['interacts']['edge_index']
    all_edges = dataset['edge_index']
    adjt = np.zeros((geneCoexpression.num_nodes,geneCoexpression.num_nodes))
    
    for i in range(all_edges.shape[1]):
        adjt[int(all_edges[0][i])][int(all_edges[1][i])] = 1
        adjt[int(all_edges[1][i])][int(all_edges[0][i])] = 1
    #print("adjt: ",adjt)
    
    all_neg_edges_x = []
    all_neg_edges_y = []
    
    for i in range(geneCoexpression.num_nodes):
        for j in range(i+1,geneCoexpression.num_nodes):
            if adjt[i][j] == 1:
                continue
            else:
                all_neg_edges_x.append(i)
                all_neg_edges_y.append(j)
    all_neg_edges = [all_neg_edges_x,all_neg_edges_y]
    #print("all_neg_edges.shape",len(all_neg_edges[0]))
    all_neg_edges = torch.LongTensor(all_neg_edges)
    #print("all_neg_edges.shape",all_neg_edges.shape)
    
    
    geneCoexpression.edge_attr = None
    
    split_edge = edge_split_direct(geneCoexpression)   
    data = geneCoexpression
    
    

    data.edge_index = to_undirected(split_edge['train']['edge'].t())
    edge_index, _ = add_self_loops(data.edge_index)             
    adj = SparseTensor.from_edge_index(edge_index).t()
    data = data.to(device)
    
    adj = adj.to(device)

    save_path_model = 'gcn'+ '_model.pth'
    save_path_predictor = 'gcn'+'_pred.pth'
    
    out_name = ''

    metric = 'AUC'
    predictor = LPD(args.hidden_channels, args.decode_channels, 1, args.num_layers,args.decode_layers, args.dropout).to(device)
                              
    model = GCN_mgae(data.num_features, args.hidden_channels,args.hidden_channels, args.num_layers,args.dropout, decoder_mask=args.decoder_mask, num_nodes=data.num_nodes).to(device)
    
        
    
    loggers = {
        'AUC': Logger(args.runs, args),
        'AUPR': Logger(args.runs, args)
    }
    

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,args)
            t2 = time.time()

            results = test(model, predictor, data, adj, split_edge, args.batch_size)
                           
            

            valid_hits = results[metric][1]
            if valid_hits > best_valid:
                best_valid = valid_hits
                best_epoch = epoch
                torch.save(model.state_dict(), save_path_model)
                torch.save(predictor.state_dict(), save_path_predictor)
                cnt_wait = 0
            else:
                cnt_wait += 1

            
            if cnt_wait == args.patience:
                #print('Early stopping!')
                break
        
        model.load_state_dict(torch.load(save_path_model))
        predictor.load_state_dict(torch.load(save_path_predictor))
        results = test(model, predictor, data, adj, split_edge,args.batch_size)
        
                       
        for key, result in results.items():
            loggers[key].add_result(run, result)

    print(f" Final Testing result- {data_name}  ")
    for key in loggers.keys():
        
        loggers[key].print_statistics(key)
        #break
        
        
   

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HyperSSL')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--decoder_mask', type=str, default='mask', help='mask | nmask') 
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--decode_layers', type=int, default=2)
    parser.add_argument('--in_channels', type=int, default=128) 
    parser.add_argument('--hidden_channels', type=int, default=128) 

    parser.add_argument('--decode_channels', type=int, default=256) 
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0001) 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--mask_ratio', type=float, default=0.7)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50,help='Use attribute or not')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--data_name', type = str, default = 'pd') #t2d, pd, hd, sch
    warnings.simplefilter('ignore')
    warnings.filterwarnings("ignore")
    
    args = parser.parse_args()
    HyperSSL('hd',args) # t2d, pd, hd, sch

