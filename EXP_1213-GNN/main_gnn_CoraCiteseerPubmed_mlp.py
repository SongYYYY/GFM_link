import sys
sys.path.append("..") 

import torch
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
from scoring import mlp_score

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch.nn as nn

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from GFM_utils import generate_link_prediction_data, reset_gnn_weights

dir_path  = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())

class LinkPredModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, dropout_rate=0.5):
        """
        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int): The dimensionality of the hidden layer.
            output_dim (int): The dimensionality of the output features.
            dropout_rate (float): The dropout probability.
        """
        super(LinkPredModel, self).__init__()

        # Define the MLP layers with dropout
        self.extractor = ResidualMLP(input_dim, hidden_dim, hidden_dim, n_layers, dropout_rate)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, pos_graph, neg_graph):
        x = self.encode(x)
        scores_pos, scores_neg = self.decode(x, pos_graph, neg_graph)

        return scores_pos, scores_neg
    
    def encode(self, x):
        return self.extractor(x)
    
    def decode(self, h, pos_graph, neg_graph):
        src_pos, dst_pos = pos_graph.edges()
        src_neg, dst_neg = neg_graph.edges()

        scores_pos = self.projector(torch.cat([h[src_pos], h[dst_pos]], dim=1)).squeeze()
        scores_neg = self.projector(torch.cat([h[src_neg], h[dst_neg]], dim=1)).squeeze()
        
        return scores_pos, scores_neg


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

class my_score(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers,
                 dropout):
        super().__init__()

        self.projector = nn.Sequential(
                    nn.Linear(hidden_dim*2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
        )

        self.dropout = dropout

    def reset_parameters(self):
        pass

    def forward(self, x_i, x_j):
        x = self.projector(torch.cat([x_i, x_j], dim=1))

        return torch.sigmoid(x)

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 3, 10, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in [1,3,10, 100]:
    #     result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

   
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    
    return result

        

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
def test_edge(score_func, input_data, h, batch_size):

    # input_data  = input_data.transpose(1, 0)
    # with torch.no_grad():
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
    
        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
        
    pred_all = torch.cat(preds, dim=0)

    return pred_all


@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    
    
    h = model(x, data['adj'].to(x.device))
    # print(h[0][:10])
    x = h

    pos_train_pred = test_edge(score_func, data['train_val'], h, batch_size)

    neg_valid_pred = test_edge(score_func, data['valid_neg'], h, batch_size)

    pos_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size)

    pos_test_pred = test_edge(score_func, data['test_pos'], h, batch_size)

    neg_test_pred = test_edge(score_func, data['test_neg'], h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]

    return result, score_emb

def load_weights(model, score_func, ckpt_path):
    state_dict = torch.load(ckpt_path)
       
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
    score_func.projector[0].weight.data = state_dict['projector.0.weight']
    score_func.projector[2].weight.data = state_dict['projector.2.weight']
    score_func.projector[4].weight.data = state_dict['projector.4.weight']
    score_func.projector[0].bias.data = state_dict['projector.0.bias']
    score_func.projector[2].bias.data = state_dict['projector.2.bias']
    score_func.projector[4].bias.data = state_dict['projector.4.bias']
    # Remove the 'projector.' prefix and assign the value to score_func
    # score_func..data = value
    # print(f'{new_key} loaded.')
    print('score_func loaded.')
    return 


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ###### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    
    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print(args)

    init_seed(1)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    downstream_root = '/local/scratch/ysong31/GFM_dev/LLM_data'
    ckpt_path = f'./ckpt_clf/papers_best.ckpt'
    data_name = args.data_name
    data_ori = torch.load(os.path.join(downstream_root, f'{data_name}_fixed_sbert.pt'))

    data = generate_link_prediction_data(data_ori, train_ratio=0.85, valid_ratio=0.05, seed=123)
    data['x'] = data_ori.x
   
    node_num = data['x'].size(0)

    x = data['x']

    if args.cat_n2v_feat:
        print('cat n2v embedding!!')
        n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name+'-n2v-embedding.pt'))
        x = torch.cat((x, n2v_emb), dim=-1)

    x = x.to(device)
    train_pos = data['train_pos'].to(x.device)

    input_channel = x.size(1)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num, args.cat_node_feat_mf).to(device)
    
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    
    # print(state_dict.keys())
    
    # model = ResidualMLP(384, 384, 384, 5, 0.5).to(device)
    # score_func = my_score(384, 3, 0).to(device)

    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs)
    }

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        # model.reset_parameters()
        # score_func.reset_parameters()
        reset_gnn_weights(model)
        reset_gnn_weights(score_func)
        # load_weights(model, score_func, ckpt_path)
        model = model.to(device)
        score_func = score_func.to(device)

        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, score_func, train_pos, x, optimizer, args.batch_size)
            # print(model.convs[0].att_src[0][0][:10])
            
            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(model, score_func, data, x, evaluator_hit, evaluator_mrr, args.batch_size)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result


                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.save:

                        save_emb(score_emb, save_path)

                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean


            
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    
    print(best_metric_valid_str +' ' +best_auc_valid_str)

    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()

   