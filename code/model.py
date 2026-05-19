import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class DGI(nn.Module):
    def __init__(
            self,
            pre_train_data: str,
            in_dim: int,
            out_dim: int,
            num_layers: int,
            dropout: float,
    ):
        super(DGI, self).__init__()
        self.pre_train_data = pre_train_data
        self.backbone = SGC(in_dim, out_dim, num_layers, dropout)

        self.semantic_attn = SemanticAttention(out_dim, out_dim // 2)

        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)

    def forward(self, x, x_shuf, edge_index):
        emb = []
        for _, e in enumerate(edge_index):
            conv_out = self.backbone(x, e)
            emb.append(conv_out.unsqueeze(-2))
        emb = torch.cat(emb, dim=-2)
        h_1 = self.semantic_attn(emb)

        c = torch.mean(h_1, 0, keepdim=False)
        c = self.sigm(c)

        emb = []
        for _, e in enumerate(edge_index):
            conv_out = self.backbone(x_shuf, e)
            emb.append(conv_out.unsqueeze(-2))
        emb = torch.cat(emb, dim=-2)
        h_2 = self.semantic_attn(emb)

        ret = self.disc(c, h_1, h_2)
        return ret

    def embed(self, x, edge_index):
        emb = []
        for _, e in enumerate(edge_index):
            conv_out = self.backbone(x, e)
            emb.append(conv_out.unsqueeze(-2))
        emb = torch.cat(emb, dim=-2)
        h_1 = self.semantic_attn(emb)

        c = torch.mean(h_1, 0, keepdim=False)
        return h_1.detach(), c.detach()


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), -1).unsqueeze(0)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), -1).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class SGC(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, dropout=0.2):
        super(SGC, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.conv = SGConv(in_dim, out_dim, K=num_layers, cached=False)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=-2)
        return (beta * z).sum(-2, keepdim=False)


class Classifier(nn.Module):
    def __init__(self, expert_out_dim, hidden_dim, nb_classes, nb_expert, dropout):
        super(Classifier, self).__init__()
        self.nb_expert = nb_expert

        self.fc = nn.ModuleList([nn.Linear(expert_out_dim, hidden_dim, bias=True) for _ in range(nb_expert)])
        self.proj = nn.Linear(hidden_dim, nb_classes, bias=False)

        self.class_attn_q = nn.Parameter(torch.randn(nb_classes, 1, hidden_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)
        self.expert = None
        self.expert_out_cache = [None for _ in range(nb_expert)]
        self.weights_init()

    def weights_init(self):
        nn.init.xavier_uniform_(self.class_attn_q)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def set_expert(self, expert):
        self.expert = expert

    def ortho_loss(self):
        M = self.class_attn_q.squeeze(1)  # (c,1,d) → (c,d)
        M_norm = F.normalize(M, p=2, dim=-1)

        gram_matrix = torch.matmul(M_norm, M_norm.T)  # (c,c)
        identity = torch.eye(gram_matrix.shape[0], device=gram_matrix.device)  # (c,c)

        ortho_loss = F.mse_loss(gram_matrix, identity, reduction='mean')
        return ortho_loss

    def forward(self, x, edge_index, idx):
        all_emb, labeled_emb = [], []
        for k in range(self.nb_expert):
            if self.expert_out_cache[k] is None:
                cur_out = self.expert[k].embed(x, edge_index)[0]
                self.expert_out_cache[k] = cur_out

            h = self.expert_out_cache[k]
            h = self.fc[k](self.dropout(h))
            h = F.relu(h)
            all_emb.append(h)
            labeled_emb.append(F.embedding(idx, h))

        all_emb = torch.cat([h.unsqueeze(0) for h in all_emb], dim=0)
        labeled_emb = torch.cat([h.unsqueeze(0) for h in labeled_emb], dim=0)

        q = self.class_attn_q
        q = F.normalize(q, p=2, dim=-1)
        labeled_emb = F.normalize(labeled_emb, p=2, dim=-1)
        sim = torch.matmul(labeled_emb, q.transpose(-1, -2))
        sim = self.dropout(sim)
        mean_sim = torch.mean(sim, dim=-2, keepdim=True)
        attention = F.softmax(mean_sim * self.temperature, dim=0)
        attention = attention.squeeze()

        all_emb = self.dropout(all_emb)
        logits = self.proj(all_emb)

        logits = logits * attention.unsqueeze(-2)
        logits = logits.sum(dim=0, keepdim=False)
        return logits, self.ortho_loss()
