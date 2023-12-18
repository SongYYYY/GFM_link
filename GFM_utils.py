from torch_geometric.utils import negative_sampling, remove_self_loops, coalesce
import numpy as np
import random
from torch_sparse import SparseTensor
import torch
import pickle

def convert_to_single_direction(edge_index):
    # Extract edges from the graph
    src, dst = edge_index[0], edge_index[1]

    # Use numpy for efficient processing
    min_edges = np.minimum(src.numpy(), dst.numpy())
    max_edges = np.maximum(src.numpy(), dst.numpy())

    # Create unique edges
    unique_edges = np.unique(np.vstack((min_edges, max_edges)), axis=1)
    # Create a new graph with these edges
    new_edge_index = torch.from_numpy(unique_edges).long()

    return new_edge_index



def add_reversed_edges(edge_index):
    # Add reversed edges to make the graph bi-directional
    reversed_edges = edge_index[[1, 0], :]
    return torch.cat([edge_index, reversed_edges], dim=1)

def generate_link_prediction_data(data, train_ratio, valid_ratio, seed=123):
    # Set seed for reproducibility in NumPy, Python random and PyTorch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Check and infer test_ratio
    test_ratio = 1 - train_ratio - valid_ratio
    if test_ratio < 0:
        raise ValueError("Sum of train_ratio and valid_ratio should be less than 1.")

    # Process the graph data: Remove self-loops and duplicate edges
    edge_index, _ = coalesce(data.edge_index, None, data.num_nodes, data.num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = convert_to_single_direction(edge_index)
    
    # Randomly sample edges for train, valid, and test
    num_edges = edge_index.size(1)
    all_indices = np.random.permutation(num_edges)
    train_end = int(num_edges * train_ratio)
    valid_end = int(num_edges * (train_ratio + valid_ratio))

    train_indices = all_indices[:train_end]
    valid_indices = all_indices[train_end:valid_end]
    test_indices = all_indices[valid_end:]

    result_data = {
        'train_pos': edge_index[:, train_indices],
        'valid_pos': edge_index[:, valid_indices],
        'test_pos': edge_index[:, test_indices],
    }

    # Add a subset of train_pos as the validation set for training edges
    result_data['train_val'] = result_data['train_pos'][:, :result_data['valid_pos'].shape[1]]

    # Make edges bi-directional for negative edge sampling
    bi_directional_train_valid = add_reversed_edges(torch.cat([result_data['train_pos'], result_data['valid_pos']], dim=1))

    # Sample negative edges for train and valid
    result_data['train_neg'] = negative_sampling(bi_directional_train_valid, num_nodes=data.num_nodes, num_neg_samples=train_indices.size)
    result_data['valid_neg'] = negative_sampling(bi_directional_train_valid, num_nodes=data.num_nodes, num_neg_samples=valid_indices.size)

    # Make all edges bi-directional including test edges for final negative edge sampling
    all_bi_directional_edges = add_reversed_edges(torch.cat([result_data['train_pos'], result_data['valid_pos'], result_data['test_pos']], dim=1))
    result_data['test_neg'] = negative_sampling(all_bi_directional_edges, num_nodes=data.num_nodes, num_neg_samples=test_indices.size)

    bi_directional_train = add_reversed_edges(result_data['train_pos'])
    edge_weight = torch.ones(bi_directional_train.shape[1])
    result_data['adj'] = SparseTensor.from_edge_index(bi_directional_train, edge_weight, [data.num_nodes, data.num_nodes])

    result_data['train_pos'] = result_data['train_pos'].t()
    result_data['train_val'] = result_data['train_val'].t()
    result_data['valid_neg'] = result_data['valid_neg'].t()
    result_data['valid_pos'] = result_data['valid_pos'].t()
    result_data['test_pos'] = result_data['test_pos'].t()
    result_data['test_neg'] = result_data['test_neg'].t()
     
    return result_data

def sample_negative_edges(pos_edge_index, num_nodes, K, observed_edges):
    # Create a tensor to hold negative edges, with shape (n, K)
    neg_edges = torch.zeros((pos_edge_index.size(1), K), dtype=torch.long)

    for i, edge in enumerate(pos_edge_index.t()):
        src_node = edge[0]
        # Mask for valid negative target nodes
        mask = torch.ones(num_nodes, dtype=torch.bool)

        # Mask out nodes where (src_node, v) or (v, src_node) exist in observed edges
        mask[observed_edges[0][observed_edges[1] == src_node]] = False
        mask[observed_edges[1][observed_edges[0] == src_node]] = False
        mask[src_node] = False  # Also mask out self-loop

        # Get valid negative targets
        valid_neg_targets = mask.nonzero(as_tuple=False).view(-1)

        # Randomly sample K nodes from the valid negative targets
        num_neg_samples = min(K, valid_neg_targets.size(0))
        if num_neg_samples > 0:
            sampled_neg_targets = valid_neg_targets[torch.randperm(valid_neg_targets.size(0))[:num_neg_samples]]
            neg_edges[i, :num_neg_samples] = sampled_neg_targets

            # If there are fewer than K valid neg targets, pad the rest with -1
            if num_neg_samples < K:
                neg_edges[i, num_neg_samples:] = -1

    return neg_edges

def generate_link_prediction_data_(data, data_name, train_ratio, valid_ratio, K, seed=42, save=True, save_dir='.'):
    save_path = os.path.join(save_dir, f'{data_name}_{train_ratio}_{valid_ratio}_{K}_{seed}.pkl')
    if os.path.exists(save_path):
        # If the file exists, then open and load it
        with open(save_path, 'rb') as pickle_file:
            result_data = pickle.load(pickle_file)
        print("Successfully loaded file.")
        return result_data
    
   # Set seed for reproducibility in NumPy, Python random and PyTorch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Check and infer test_ratio
    test_ratio = 1 - train_ratio - valid_ratio
    if test_ratio < 0:
        raise ValueError("Sum of train_ratio and valid_ratio should be less than 1.")

    # Process the graph data: Remove self-loops and duplicate edges
    edge_index, _ = coalesce(data.edge_index, None, data.num_nodes, data.num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = convert_to_single_direction(edge_index)
    
    # Randomly sample edges for train, valid, and test
    num_edges = edge_index.size(1)
    all_indices = np.random.permutation(num_edges)
    train_end = int(num_edges * train_ratio)
    valid_end = int(num_edges * (train_ratio + valid_ratio))

    train_indices = all_indices[:train_end]
    valid_indices = all_indices[train_end:valid_end]
    test_indices = all_indices[valid_end:]

    result_data = {
        'train_pos': edge_index[:, train_indices],
        'valid_pos': edge_index[:, valid_indices],
        'test_pos': edge_index[:, test_indices],
    }

    # Combine all positive edges to create a set of observed edges
    valid_observed_edges = torch.cat([result_data['train_pos'], result_data['valid_pos']], dim=1)
    test_observed_edges = torch.cat([result_data['train_pos'], result_data['valid_pos'], result_data['test_pos']], dim=1)
    
    # Sample negative edges for valid and test
    result_data['valid_neg'] = sample_negative_edges(result_data['valid_pos'], data.num_nodes, K, valid_observed_edges)
    result_data['test_neg'] = sample_negative_edges(result_data['test_pos'], data.num_nodes, K, test_observed_edges)

    bi_directional_train = add_reversed_edges(result_data['train_pos'])
    edge_weight = torch.ones(bi_directional_train.shape[1])
    result_data['adj'] = SparseTensor.from_edge_index(bi_directional_train, edge_weight, [data.num_nodes, data.num_nodes])

    result_data['train_pos'] = result_data['train_pos'].t()
    result_data['valid_pos'] = result_data['valid_pos'].t()
    result_data['test_pos'] = result_data['test_pos'].t()

    if save:
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(result_data, pickle_file)
        print(f'Data saved to {save_path}.')

    return result_data

# Usage example
# result_data = generate_link_prediction_data(data, train_ratio=0.7, valid_ratio=0.2, K=5, seed=123)


def reset_gnn_weights(model):
    for m in model.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


def set_random_seed(seed):
    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python
    random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

import os
def set_seed_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True