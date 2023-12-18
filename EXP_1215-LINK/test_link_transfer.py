import sys
sys.path.append("..") 

import torch
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
from scoring import SymmetricScoringFunction, AsymmetricScoringFunction

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch.nn as nn

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from GFM_utils import generate_link_prediction_data_, reset_gnn_weights, set_random_seed
import tqdm
import fitlog
from copy import deepcopy

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout):
        super(ResidualMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_t):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = F.relu(layer(x))
            else:
                identity = x  # Save input for residual connection
                x = layer(x)
                x = F.relu(x)
                x = self.dropout(x)
                x = x + identity  # Add residual connection

        identity = x
        x = self.output_layer(x)
        x = F.relu(x)
        x = x + identity

        return x


def train(model, score_func, train_pos, x, optimizer, batch_size):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()


        num_nodes = x.size(0)

        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
    
        train_edge_mask = train_pos[mask].transpose(1,0)

        # train_edge_mask = to_undirected(train_edge_mask)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        # edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
            
        ###################
        # print(adj)

        h = model(x, adj)

        edge = train_pos[perm].t()

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        if isinstance(score_func, AsymmetricScoringFunction):
            pos_out = score_func(h[edge[1]], h[edge[0]])
            pos_loss += -torch.log(pos_out + 1e-15).mean()
            pos_loss = pos_loss / 2

        # Just do some trivial random sampling.
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def compute_mrr(
    score_func, evaluator, node_emb, src, dst, neg_dst, device, batch_size=2048
):
    """Compute Mean Reciprocal Rank (MRR) in batches."""
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc="Evaluate"):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device) # (N, 1, d)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device) # (N, K+1, d)

        pred = score_func(h_src.expand(h_dst.shape[0], h_dst.shape[1], h_src.shape[2]), h_dst).squeeze() # (N, K+1)
        input_dict = {"y_pred_pos": pred[:, 0], "y_pred_neg": pred[:, 1:]}
        rr[start:end] = evaluator.eval(input_dict)["mrr_list"]
    return rr.mean()


def evaluate_mrr(device, data, model, score_func, split='test'):
    model.eval()
    score_func.eval()
    evaluator = Evaluator(name="ogbl-citation2")
    with torch.no_grad():
        node_emb = model(data['x'].to(device), data['adj'].to(device))
        src = data[f'{split}_pos'].t()[0].to(node_emb.device) # test_pos: [n, 2] -> src: [n]
        dst = data[f'{split}_pos'].t()[1].to(node_emb.device) # dst: [n]
        neg_dst = data[f'{split}_neg'].to(node_emb.device) # test_neg: [n, K]
        mrr = compute_mrr(
                score_func, evaluator, node_emb, src, dst, neg_dst, device
            )
    return mrr

def get_model(gnn_model, score_model, input_channel, hidden_channels, gnn_layers, score_layers, dropout, head):
    if gnn_model == 'GCN':
        model = GCN(input_channel, hidden_channels, hidden_channels, gnn_layers, dropout, None, None, None, False, None)
    elif gnn_model == 'GAT':
        model = GAT(input_channel, hidden_channels, hidden_channels, gnn_layers, dropout, None, head, None, False, None)
    elif gnn_model == 'SAGE':
        model = SAGE(input_channel, hidden_channels, hidden_channels, gnn_layers, dropout, None, None, None, False, None)
    elif gnn_model == 'MLP':
        model = MLP(input_channel, hidden_channels, hidden_channels, gnn_layers, dropout, None, None, None, False, None)
    else:
        raise TypeError(f'Unrecognized GNN Type! {gnn_model}')
    
    if score_model == 'Sym':
        score_func = SymmetricScoringFunction(hidden_channels, hidden_channels, 1, score_layers, dropout)
    elif score_model == 'Asym':
        score_func = AsymmetricScoringFunction(hidden_channels, hidden_channels, 1, score_layers, dropout)
    else:
        raise TypeError(f'Unrecognized Score Function! {score_model}')
    
    return model, score_func



def load_weights(model, score_func, ckpt_path):
    state_dict = torch.load(ckpt_path)
    # print(state_dict.keys())
    model.layers[0].weight.data = state_dict['extractor.layers.0.weight']
    model.layers[1].weight.data = state_dict['extractor.layers.1.weight']
    model.layers[2].weight.data = state_dict['extractor.layers.2.weight']
    model.layers[3].weight.data = state_dict['extractor.layers.3.weight']
    model.layers[0].bias.data = state_dict['extractor.layers.0.bias']
    model.layers[1].bias.data = state_dict['extractor.layers.1.bias']
    model.layers[2].bias.data = state_dict['extractor.layers.2.bias']
    model.layers[3].bias.data = state_dict['extractor.layers.3.bias']
    model.output_layer.weight.data = state_dict['extractor.output_layer.weight']
    model.output_layer.bias.data = state_dict['extractor.output_layer.bias']
    print('model loaded.')
    if isinstance(score_func, AsymmetricScoringFunction):
        score_func.lins[0].weight.data = state_dict['projector.0.weight']
        score_func.lins[1].weight.data = state_dict['projector.2.weight']
        score_func.lins[2].weight.data = state_dict['projector.4.weight']
        score_func.lins[0].bias.data = state_dict['projector.0.bias']
        score_func.lins[1].bias.data = state_dict['projector.2.bias']
        score_func.lins[2].bias.data = state_dict['projector.4.bias']
        print('score_func loaded.')
        # Remove the 'projector.' prefix and assign the value to score_func
        # score_func..data = value
        # print(f'{new_key} loaded.')
    #     print('score_func loaded.')
    # elif isinstance(score_func, SymmetricScoringFunction):
    #     model.layers[0].weight.data = state_dict['extractor.layers.0.weight']
    #     model.layers[0].bias.data = state_dict['extractor.layers.0.bias']
    #     try:
    #         model.layers[1].weight.data = state_dict['extractor.layers.1.weight']
    #         model.layers[1].bias.data = state_dict['extractor.layers.1.bias']
    #         print('LOADED!')
    #     except:
    #         print('EXCEPTION!')
    #     model.output_layer.weight.data = state_dict['extractor.output_layer.weight']
    #     model.output_layer.bias.data = state_dict['extractor.output_layer.bias']
    #     print('model loaded.')
    #     score_func.lins[0].weight.data = state_dict['projector.0.weight']
    #     score_func.lins[1].weight.data = state_dict['projector.3.weight']
    #     score_func.lins[2].weight.data = state_dict['projector.6.weight']
    #     score_func.lins[0].bias.data = state_dict['projector.0.bias']
    #     score_func.lins[1].bias.data = state_dict['projector.3.bias']
    #     score_func.lins[2].bias.data = state_dict['projector.6.bias']
    #     print('score_func loaded.')
    return 

def add_hyper(params):
    tmp_dict={}
    for k,v in params.items():
        if isinstance(v,tuple) or isinstance(v,list):
            tmp_dict[k]=v
    fitlog.add_hyper(params)
    for k,v in tmp_dict.items():
        params[k]=v

def init_fitlog(param_grid, log_dir='logs'):
    fitlog.commit(__file__)
    fitlog.set_log_dir(log_dir)
    add_hyper(param_grid)

def train_and_eval(param_grid):
    gpu_id = param_grid['gpu']
    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
    print(device)

    init_fitlog(param_grid, 'logs_gnn')
    print('Fitlog init.')

    seed = param_grid['seed']
    set_random_seed(seed)

    # data 
    data_name = param_grid['data_name']
    train_ratio = param_grid['train_ratio']
    val_ratio = param_grid['val_ratio']
    data_seed = param_grid['data_seed']
    emb_type = param_grid['emb_type']
    # model
    gnn_model = param_grid['gnn_model']
    score_model = param_grid['score_model']
    gnn_layers = param_grid['gnn_layers']
    score_layers = param_grid['score_layers']
    hidden_dim = param_grid['hidden_dim']
    dropout = param_grid['dropout']
    # train
    finetune = param_grid['finetune']
    ckpt_name = param_grid['ckpt_name']
    batch_size = param_grid['batch_size']
    lr = param_grid['lr']
    epochs = param_grid['epochs']
    eval_steps = param_grid['eval_steps']
    runs = param_grid['runs']
    patience = param_grid['patience']
    l2 = param_grid['l2']
    ######gat
    head = param_grid['head'] 
    # evaluation
    K = param_grid['K']

    set_random_seed(seed)

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    downstream_root = '/local/scratch/ysong31/GFM_dev/LLM_data'
    ckpt_path = f'./{ckpt_name}/papers_best.ckpt'
    data_ori = torch.load(os.path.join(downstream_root, f'{data_name}_fixed_{emb_type}.pt'))

    data = generate_link_prediction_data_(data_ori, data_name, train_ratio=train_ratio, valid_ratio=val_ratio, K=K, seed=data_seed,
                                          save=True, save_dir='../link_data')
    data['x'] = data_ori.x
   
    node_num = data['x'].size(0)
    x = data['x']
    x = x.to(device)
    train_pos = data['train_pos'].to(x.device)
    input_channel = x.size(1)

    # model, score_func = get_model(gnn_model, score_model, input_channel, hidden_dim, gnn_layers, score_layers, dropout, head)
    model = ResidualMLP(384, 384, 384, 5, dropout)
    if score_model == 'Asym':
        score_func = AsymmetricScoringFunction(384, 384, 1, score_layers, dropout)
    else:
        score_func = SymmetricScoringFunction(384, 384, 1, score_layers, dropout)
    model = model.to(device)
    score_func = score_func.to(device)

    runs = 1 if finetune else runs
    mrr_list = []
    epoch_list = []
    for run in range(runs):

        print('#################################          ', run, '          #################################')
        
        if runs == 1:
            train_seed = seed
        else:
            train_seed = run
        print('train seed: ', train_seed)

        set_random_seed(train_seed)

        if finetune:
            load_weights(model, score_func, ckpt_path)
        else:
            reset_gnn_weights(model)
            reset_gnn_weights(score_func)

        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()), lr=lr, weight_decay=l2)

        best_valid = 0
        best_epoch = -1
        count = 0
        for epoch in range(1, 1 + epochs):
            loss = train(model, score_func, train_pos, x, optimizer, batch_size)
            
            if epoch % eval_steps == 0:
                mrr = evaluate_mrr(device, data, model, score_func, 'valid')
                print('Valid MRR: {:.4f}'.format(mrr))

                if best_valid < mrr:
                    best_valid = mrr
                    best_epoch = epoch
                    count = 0
                    model_weights = deepcopy(model.state_dict())
                    score_weights = deepcopy(score_func.state_dict())
                else:
                    count += 1
                    if count >= patience:
                        break 

        print('RUN: {}, Training Stop! Best MRR: {:.4f} at Epoch {}'.format(run, best_valid, best_epoch))
        model.load_state_dict(model_weights)
        score_func.load_state_dict(score_weights)
        print('Ckpt loaded.')
        test_mrr = evaluate_mrr(device, data, model, score_func, 'test')
        mrr_list.append(test_mrr)
        print('RUN: {}, TEST MRR: {:.4f}'.format(run, test_mrr))
        epoch_list.append(best_epoch)

    print('TEST MRR LIST: {}'.format(mrr_list))
    print('MEAN TEST MRR: {:.4f}, STD: {:.4f}, Epoch: {}'.format(np.mean(mrr_list), np.std(mrr_list), np.mean(epoch_list)))
    fitlog.add_best_metric({'MEAN': np.mean(mrr_list), 'STD': np.std(mrr_list), 'EPOCH': np.mean(epoch_list)}, name='TEST-MRR')


import time
def train_and_record(pId, param_grid):
    try:
        train_and_eval(param_grid)
    except Exception as e:
        print('error occured in {}.'.format(param_grid))
        print(e)
        fitlog.finish()
        # record errors
        with open('error_log.txt', 'a') as f:
            f.writelines([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), str(param_grid), str(e)])
            f.write('\n')
    # train_and_eval(param_grid)
    return

if __name__ == '__main__':
    param_grid = {
    # data
    'data_name': 'cora',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'data_seed': 0,
    'emb_type': 'sbert',
    # model 
    'gnn_model': 'MLP',
    'score_model': 'Asym',
    'gnn_layers': 5,
    'score_layers': 3,
    'hidden_dim': 384,
    'dropout': 0.5,
    # train
    'finetune': True,
    'ckpt_name': 'ckpt_clf',
    'batch_size': 4096,
    'lr': 1e-3,
    'l2': 0,
    'epochs': 1000,
    'patience': 20,
    'runs': 3,
    'eval_steps': 5,
    # gat
    'head': 1,
    # evaluation
    'K': 100,
    # others
    'gpu': 0,
    'seed': 123,
    }

    train_and_eval(param_grid)
