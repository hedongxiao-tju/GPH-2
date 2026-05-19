import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, add_self_loops

from model import DGI, LogReg
from data import load_data_homo, load_data_het
from utils import set_all_seed, create_few_data_folder, load_few_shot_split, accuracy


def pre_train():
    for data in datasets:
        model_save_path = f"./../checkpoint/{data}.pth"
        if data in ["ACM", "Aminer", "DBLP", "Freebase"]:
            x, edge_index, y, num_nodes, _, num_features, num_classes = load_data_het(
                data_dir, data, decomposed_dim)
            x = x.to(device)
            edge_index = [e.to(device) for e in edge_index]
        elif data in ["cora", "citeseer", "pubmed", "amazon-photo", "amazon-computer"]:
            x, edge_index, y, num_nodes, _, num_features, num_classes = load_data_homo(
                data_dir, data, decomposed_dim)
            x = x.to(device)
            edge_index = [edge_index.to(device)]
        else:
            raise Exception("Unknown dataset")

        model = DGI(
            pre_train_data=data,
            in_dim=num_features,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        best_loss = float('inf')
        best_epoch = None
        patience = 0
        print('Training ...')
        for epoch in range(num_epochs):
            idx = np.random.permutation(num_nodes)
            x_shuffle = x[idx, :]

            if len(edge_index) == 1:
                edge_index_full = edge_index[0].to(device)
                for _ in range(num_homo_view):
                    dropped_edge_index, _ = dropout_edge(edge_index_full, p=drop_edge_rate, force_undirected=True)
                    edge_index.append(add_self_loops(dropped_edge_index)[0])
                edge_index = edge_index[:-1]

            lbl_1 = torch.ones(1, num_nodes).to(device)
            lbl_2 = torch.zeros(1, num_nodes).to(device)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            model.train()
            optimiser.zero_grad()

            logits = model(x, x_shuffle, edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, lbl)

            loss.backward()
            optimiser.step()
            print('{:<10} epoch:{:<5} loss:{:.6f}'.format(data, epoch, loss.item()))

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                patience = 0
                torch.save(model, model_save_path)
            else:
                patience += 1

            if patience >= 50:
                print(f'patience: {patience}! early stopping...')
                break

        mean_acc = indomain_few_shot_eval(
            model_save_path, data, x, edge_index, y, num_classes, device, num_shots=3, num_tasks=10)
        summary = '{:<10} best_epoch:{} best_loss:{:.6f} few_shot_acc:{:.6f}'.format(
            data, best_epoch, best_loss, mean_acc)
        print(summary)


def indomain_few_shot_eval(model_path, data_name, x, edge_index, y, num_classes, device, num_shots, num_tasks):
    model = torch.load(model_path, weights_only=False)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out, _ = model.embed(x, edge_index)

    in_dim = out.shape[-1]
    create_few_data_folder(num_shots, num_tasks, y, data_dir, data_name)

    mean_acc = []
    for task_num in range(1, num_tasks + 1):
        model = LogReg(in_dim, num_classes).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        train_idx, train_lbls, test_idx, test_lbls = load_few_shot_split(
            num_shots, task_num, data_dir, data_name, device)

        best_loss, best_epoch, patience = float('inf'), 0, 0
        best_model_state = None
        for epoch in range(1000):
            model.train()
            optimiser.zero_grad()

            logits = model(out)
            train_loss = F.cross_entropy(logits[train_idx], train_lbls)
            train_loss.backward()
            optimiser.step()

            if train_loss < best_loss:
                best_loss = train_loss
                patience = 0
                best_model_state = model.state_dict()
            else:
                patience += 1
            if patience >= 20:
                break

        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            logits = model(out)
            test_acc = accuracy(logits[test_idx].detach(), test_lbls)
            mean_acc.append(test_acc.cpu().numpy())

    return np.mean(mean_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", default=False)

    parser.add_argument("--data_dir", type=str, default=r'./../data')
    parser.add_argument("--dataset", type=str, default="cora_citeseer_pubmed_amazon-photo_amazon-computer", help="All pre-trained datasets, split by '_'")

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--decomposed_dim", type=int, default=128)
    parser.add_argument("--num_homo_view", type=int, default=3)
    parser.add_argument("--drop_edge_rate", type=float, default=0.3)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)

    args = parser.parse_args()
    print(args, '\n')

    set_all_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(device)

    data_dir = args.data_dir
    datasets = args.dataset.strip().split('_')
    num_epochs = args.num_epochs
    lr, wd = args.lr, args.weight_decay
    num_homo_view, drop_edge_rate = args.num_homo_view, args.drop_edge_rate
    num_layers = args.num_layers
    dropout = args.dropout
    decomposed_dim, hidden_dim, out_dim = args.decomposed_dim, args.hidden_dim, args.out_dim

    pre_train()
