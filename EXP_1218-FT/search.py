import subprocess
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pickle
import torch
from torch.multiprocessing import spawn
from test_link import train_and_record


basic_params = {'batch_size': [4096], 'lr': [1e-3, 1e-2], 'l2': [0, 1e-5], 'epochs': [1000], 'patience': [20], 'runs': [3],
                'eval_steps': [10], 'head': [1], 'K': [100]}


dataset_params_list = [ 
                        {'data_name': ['cora'], 'train_ratio': [0.8, 0.3], 'val_ratio': [0.1], 'emb_type': ['sbert', 'tfidf'], 
                             'data_seed': [0],
                        },              
                        {'dataname': ['pubmed'], 'train_ratio': [0.8, 0.3], 'val_ratio': [0.1], 'emb_type': ['sbert', 'tfidf'], 
                             'data_seed': [0],
                        },                 
] 


model_params_list = [{'gnn_model': ['MLP'], 'score_model': ['Asym', 'Sym'],
                      'gnn_layers': [1, 2, 3], 'score_layers': [1, 2, 3], 'hidden_dim': [128, 256], 'dropout': [0.1, 0.5]}]

other_params_list = [{'seed': [123], 'gpu': [0]}]

param_grid_list = []
for dataset_params in dataset_params_list:
    params_tmp = basic_params.copy()
    params_tmp.update(dataset_params)
    for model_params in model_params_list:
        param_grids = params_tmp.copy()
        param_grids.update(model_params)
        for other_params in other_params_list:
            param_grids_2 = param_grids.copy()
            param_grids_2.update(other_params)
            for grid in ParameterGrid(param_grids_2):
                param_grid_list.append(grid)

print('{} param grids in total.'.format(len(param_grid_list)))

if __name__ == '__main__':
    param_grid_list = []
    for dataset_params in dataset_params_list:
        params_tmp = basic_params.copy()
        params_tmp.update(dataset_params)
        for model_params in model_params_list:
            param_grids = params_tmp.copy()
            param_grids.update(model_params)
            for other_params in other_params_list:
                param_grids_2 = param_grids.copy()
                param_grids_2.update(other_params)
                for grid in ParameterGrid(param_grids_2):
                    param_grid_list.append(grid)

    print('{} param grids in total.'.format(len(param_grid_list)))


    for param_grid in tqdm(param_grid_list):
        print(param_grid)
    #         p = Process(target=train_and_record_2, args=(param_grid,))
    #         p.start()
    #         p.join()
        spawn(train_and_record, args=(param_grid,))
        torch.cuda.empty_cache()