import os
import scipy
import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
from torch_geometric.datasets import Planetoid, Amazon
from sklearn.preprocessing import normalize

from utils import decomposition


def load_data_homo(data_dir, data_name, decomposed_dim=-1, decomposed_method='SVD'):
    print('Dataloader: Loading Dataset', data_name)
    transform = T.NormalizeFeatures()
    if data_name == 'amazon-photo':
        dataset = Amazon(root=f'{data_dir}/Amazon', name='Photo', transform=transform)
    elif data_name == 'amazon-computer':
        dataset = Amazon(root=f'{data_dir}/Amazon', name='Computers', transform=transform)
    elif data_name in ('cora', 'citeseer', 'pubmed'):
        dataset = Planetoid(root=f'{data_dir}/Planetoid', name=data_name, transform=transform)
    else:
        raise ValueError(f'Invalid dataname {data_name}')

    if decomposed_dim > 0:
        x = dataset[0].x
        x = decomposition(x, decomposed_dim, method=decomposed_method)
        x = torch.from_numpy(x).to(torch.float32)
    else:
        x = dataset[0].x.to(torch.float32)

    y = dataset[0].y.to(torch.long)
    edge_index = dataset[0].edge_index

    edge_index = to_undirected(edge_index)
    edge_index = remove_self_loops(edge_index)[0]
    edge_index = add_self_loops(edge_index)[0]

    num_nodes = y.shape[0]
    num_edges = edge_index.shape[1]
    num_features = x.shape[1]
    num_classes = dataset.num_classes

    print(f'\tnum_nodes: {num_nodes}')
    print(f'\tnum_edges: {num_edges}')
    print(f'\tnum_features: {num_features}')
    print(f'\tnum_classes:{num_classes}')
    print(f'Dataloader: {data_name} Loading Success')
    return x, edge_index, y, num_nodes, num_edges, num_features, num_classes


def load_data_het(data_dir, data_name, decomposed_dim=-1, decomposed_method='SVD'):
    print('Dataloader: Loading Dataset', data_name)
    path = os.path.join(data_dir, data_name)

    if data_name == 'ACM':
        pap = scipy.sparse.load_npz(path + "/pap.npz").toarray()
        psp = scipy.sparse.load_npz(path + "/psp.npz").toarray()
        adjs = [pap, psp]
        features = scipy.sparse.load_npz(path + "/p_feat.npz").toarray()
        labels = np.load(path + "/labels.npy").astype('int32')
    elif data_name == 'DBLP':
        apa = scipy.sparse.load_npz(path + "/apa.npz").toarray()
        apcpa = scipy.sparse.load_npz(path + "/apcpa.npz").toarray()
        aptpa = scipy.sparse.load_npz(path + "/aptpa.npz").toarray()
        adjs = [apa, apcpa, aptpa]
        features = scipy.sparse.load_npz(path + "/a_feat.npz").astype("float32").toarray()
        labels = np.load(path + "/labels.npy").astype('int32')
    elif data_name == 'Aminer':
        pap = scipy.sparse.load_npz(path + "/pap.npz").toarray()
        prp = scipy.sparse.load_npz(path + "/prp.npz").toarray()
        adjs = [pap, prp]
        features = np.eye(pap.shape[0]).astype("float32")
        labels = np.load(path + "/labels.npy").astype('int32')
    elif data_name == 'Freebase':
        mam = scipy.sparse.load_npz(path + "/mam.npz").toarray()
        mdm = scipy.sparse.load_npz(path + "/mdm.npz").toarray()
        mwm = scipy.sparse.load_npz(path + "/mwm.npz").toarray()
        adjs = [mam, mdm, mwm]
        features = np.eye(mam.shape[0]).astype("float32")
        labels = np.load(path + "/labels.npy").astype('int32')
    else:
        raise ValueError('Invalid dataname {}'.format(data_name))

    features = normalize(features, norm='l1', axis=1)
    if decomposed_dim > 0:
        features = decomposition(features, decomposed_dim, method=decomposed_method)

    x = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    edge_index_list = []
    for adj in adjs:
        non_zero_coords = np.nonzero(adj)
        edge_index = torch.tensor(np.vstack(non_zero_coords), dtype=torch.long)
        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]
        edge_index = add_self_loops(edge_index)[0]
        edge_index_list.append(edge_index)

    num_nodes = y.shape[0]
    num_edges = [e.shape[1] for e in edge_index_list]
    num_features = x.shape[1]
    num_classes = labels.max() + 1

    print(f'\tnum_nodes: {num_nodes}')
    print(f'\tnum_edges: {num_edges}')
    print(f'\tnum_features: {num_features}')
    print(f'\tnum_classes:{num_classes}')
    print(f'Dataloader: {data_name} Loading Success')
    return x, edge_index_list, y, num_nodes, num_edges, num_features, num_classes


if __name__ == '__main__':
    data_dir = r'./../data'
    data_name_he = ['DBLP', 'ACM', 'Aminer', 'Freebase']
    data_name_ho = ['cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computer']
    decomposed_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for data_name in data_name_ho:
        x, edge_index, y, num_nodes, num_edges, num_features, num_classes, = (
            load_data_homo(data_dir, data_name, decomposed_dim=-1))
        print(x.shape, edge_index.shape, y.shape, num_nodes, num_edges, num_features, num_classes)
    for data_name in data_name_he:
        x, edge_index_list, y, num_nodes, num_edges, num_features, num_classes, = (
            load_data_het(data_dir, data_name, decomposed_dim=-1))
        print(x.shape, edge_index_list[0].shape, y.shape, num_nodes, num_edges, num_features, num_classes)
