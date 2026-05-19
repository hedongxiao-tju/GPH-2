import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

from data import load_data_homo, load_data_het
from utils import set_all_seed, create_few_data_folder, load_few_shot_split, reshape_idx_to_2d
from model import Classifier


def down_prompt(num_shots, num_tasks):
    create_few_data_folder(num_shots, num_tasks, y, data_dir, data_name)

    mean_f1_macro, mean_f1_micro = [], []
    for task_num in range(1, num_tasks + 1):
        pre_train_model = []
        for k, encoder in enumerate(encoders):
            m = torch.load(f"./../checkpoint/{encoder}.pth", weights_only=False, map_location=device)
            if num_layers != -1:
                m.backbone.conv.K = num_layers
            pre_train_model.append(m.to(device).eval())

        model = Classifier(
            expert_out_dim=encoder_out_dim,
            hidden_dim=hidden_dim,
            nb_classes=num_classes,
            nb_expert=len(encoders),
            dropout=dropout)

        model.set_expert(pre_train_model)
        model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        train_idx, train_lbls, test_idx, test_lbls = load_few_shot_split(
            num_shots, task_num, data_dir, data_name, device)
        train_idx_2d = reshape_idx_to_2d(train_idx, train_lbls, num_classes)

        best_loss, best_epoch, patience = float('inf'), 0, 0
        best_model_state = None
        # print(f'Training {task_num}th ...')
        for epoch in range(num_epochs):
            model.train()
            optimiser.zero_grad()

            logits, ortho_loss = model(x, edge_index, train_idx_2d)
            ce_loss = F.cross_entropy(logits[train_idx], train_lbls)
            train_loss = ce_loss + ortho_loss
            train_loss.backward()
            optimiser.step()

            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                patience = 0
                best_model_state = model.state_dict()
            else:
                patience += 1
            if patience >= 20:
                # print(f'patience: {patience}! early stopping...')
                break

        # print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, edge_index, train_idx_2d)
            preds = torch.argmax(logits[test_idx].detach(), dim=1)
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
            # print(f'Acc: {test_f1_micro}')
            mean_f1_macro.append(test_f1_macro*100)
            mean_f1_micro.append(test_f1_micro*100)

    f1_macro = float(np.round(np.mean(mean_f1_macro), 2))
    f1_micro = float(np.round(np.mean(mean_f1_micro), 2))
    var_f1_macro = float(np.round(np.std(mean_f1_macro), 2))
    var_f1_micro = float(np.round(np.std(mean_f1_micro), 2))
    # print(f'\nAverage acc: {f1_micro}')
    return f1_micro, var_f1_micro, f1_macro, var_f1_macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", default=False, help="use cpu")

    parser.add_argument("--data_dir", type=str, default=r'./../data')
    parser.add_argument("--data_name", type=str, default="cora")
    parser.add_argument("--encoders", type=str, default="citeseer_pubmed_amazon-photo_amazon-computer", help="Pre-trained encoders, split by '_'")

    parser.add_argument("--num_epochs", type=int, default=200, help="number of down-stream epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")

    parser.add_argument("--decomposed_dim", type=int, default=128, help="decomposed_dim")
    parser.add_argument("--num_homo_view", type=int, default=3, help="num_homo_view")
    parser.add_argument("--drop_edge_rate", type=float, default=0.3, help="drop_edge_rate")

    parser.add_argument("--encoder_out_dim", type=int, default=128, help="")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden_dim")
    parser.add_argument("--num_layers", type=int, default=5, help="num_layers")

    args = parser.parse_args()
    set_all_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(args)
    print(device)

    data_dir = args.data_dir
    encoders = args.encoders.strip().split('_')
    data_name = args.data_name

    num_epochs, lr, wd = args.num_epochs, args.lr, args.weight_decay
    dropout = args.dropout
    decomposed_dim = args.decomposed_dim

    num_layers = args.num_layers
    encoder_out_dim, hidden_dim = args.encoder_out_dim, args.hidden_dim

    if data_name in ["ACM", "Aminer", "DBLP", "Freebase"]:
        x, edge_index, y, num_nodes, num_edges, num_features, num_classes, = load_data_het(
            data_dir, data_name, decomposed_dim)
        x = x.to(device)
        edge_index = [e.to(device) for e in edge_index]
    elif data_name in [ "cora", "citeseer", "pubmed", "amazon-photo", "amazon-computer"]:
        x, edge_index, y, num_nodes, num_edges, num_features, num_classes, = load_data_homo(
            data_dir, data_name, decomposed_dim)
        x = x.to(device)
        edge_index = [edge_index.to(device)]
    else:
        raise Exception("Unknown dataset")

    f1_micro, var_f1_micro, f1_macro, var_f1_macro = down_prompt(3, 10)
    print('3-shot: micro-f1 {:.2f} ~ {:.2f} | macro-f1 {:.2f} ~ {:.2f}'.format(
        f1_micro, var_f1_micro, f1_macro, var_f1_macro))
    f1_micro, var_f1_micro, f1_macro, var_f1_macro = down_prompt(5, 10)
    print('5-shot: micro-f1 {:.2f} ~ {:.2f} | macro-f1 {:.2f} ~ {:.2f}'.format(
        f1_micro, var_f1_micro, f1_macro, var_f1_macro))
