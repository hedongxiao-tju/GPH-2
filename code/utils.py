import os
import random
import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def reshape_idx_to_2d(idx, lbls, c):
    # print(idx.shape, c)
    m = int(lbls.shape[0] / c)
    cm_idx = torch.full((c, m), -1, dtype=idx.dtype, device=idx.device)
    for cls in range(c):
        cls_mask = (lbls == cls)
        cls_idx = idx[cls_mask]
        cm_idx[cls] = cls_idx

    return cm_idx


def decomposition(raw_feature, out_dim, method='SVD'):
    if method == 'PCA':
        pca = PCA(n_components=out_dim)
        x = pca.fit_transform(raw_feature)
    elif method == 'SVD':
        svd = TruncatedSVD(n_components=out_dim)
        x = svd.fit_transform(raw_feature)
    else:
        raise ValueError('Invalid method')
    return x


def create_few_data_folder(k, tasks, labels, dataset_dir, data_name):
    for task_index in range(1, tasks + 1):
        k_shot_folder = dataset_dir + '/data_split/' + data_name + '/' + str(k) + '_shot'
        os.makedirs(k_shot_folder, exist_ok=True)

        folder = os.path.join(k_shot_folder, str(task_index))
        if not os.path.exists(folder):
            os.makedirs(folder)
            node_sample_and_save(labels, k, folder, labels.max() + 1)
            print(str(k) + ' shot ' + str(task_index) + ' th is saved!!')


def node_sample_and_save(labels, k, folder, num_classes):
    labels = labels.to('cpu')
    num_nodes = labels.shape[0]

    num_test = int(0.9 * num_nodes)
    if num_test < 1000:
        num_test = int(0.7 * num_nodes)
    test_idx = torch.randperm(num_nodes)[:num_test]
    test_labels = labels[test_idx]

    if k <= 10:
        remaining_idx = torch.randperm(num_nodes)[num_test:]
        remaining_labels = labels[remaining_idx]
    else:
        remaining_idx = torch.randperm(num_nodes)
        remaining_labels = labels[remaining_idx]

    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
    shuffled_indices = torch.randperm(train_idx.size(0))
    train_idx = train_idx[shuffled_indices]
    train_labels = labels[train_idx]

    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))


def load_few_shot_split(k, task_num, dataset_dir, data_name, device):
    k_shot_folder = dataset_dir + '/data_split/' + data_name + '/' + str(k) + '_shot'
    train_idx = torch.load(k_shot_folder + "/{}/train_idx.pt".format(task_num), weights_only=False).type(torch.long).to(
        device)
    train_lbls = torch.load(k_shot_folder + "/{}/train_labels.pt".format(task_num), weights_only=False).type(
        torch.long).squeeze().to(device)
    test_idx = torch.load(k_shot_folder + "/{}/test_idx.pt".format(task_num), weights_only=False).type(torch.long).to(
        device)
    test_lbls = torch.load(k_shot_folder + "/{}/test_labels.pt".format(task_num), weights_only=False).type(
        torch.long).squeeze().to(device)
    return train_idx, train_lbls, test_idx, test_lbls
