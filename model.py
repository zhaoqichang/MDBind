import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# Custom function
from conv_util import *


# from config import *

def setALlSeed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


# ---------------Graph----------------

class EdgeFeature(nn.Module):
    def __init__(self, num_hidden=64, rbf_num=16, top_k=5, D_max=20.):
        super(EdgeFeature, self).__init__()
        self.top_k = top_k
        self.rbf_num = rbf_num
        self.D_max = D_max
        self.edge_emb = nn.Linear(rbf_num * 6 + 25, num_hidden)
        self.norm_edge = nn.LayerNorm(num_hidden)

        # [修改] 增加 2 维用于拼接 chi1 角度 (sin, cos)，27 + 2 = 29
        self.node_emb = nn.Linear(rbf_num * 15 + 29, num_hidden)
        self.norm_node = nn.LayerNorm(num_hidden)

    def forward(self, xyz, mask, angles):  # [修改] 参数列表中新增 angles [B, N, 2]
        # 根据xyz生成节点特征 N-CA CA-C CA-R CA-O
        #                  0  1  1  2 1 4 2  3
        CaX = xyz[:, :, 1]  # [B, N, 3]
        edge_index = self._distance(CaX, mask)

        node_angle = self._node_angle(xyz, mask)  # [B,N,12]
        # N-Ca Ca-R Ca-C C-O
        node_dir, edge_dir = self._node_direct(xyz, edge_index)  # [B,N,15]

        node_rbf = self._node_rbf(xyz)  # [B,N, rbf_num*15]
        if angles.dim() == 4:
            angles = angles.squeeze(-2)
        # [修改] 将外部传入的侧链 angles 拼接入纯几何特征流
        geo_node_feat = torch.cat([node_dir, node_angle, node_rbf, angles],
                                  dim=-1)  # [B, N, 15 + 12 + rbf_num * 15 + 2]

        # edge, edge_index = self._edge_feature(xyz, mask)
        edge_rbf = self._edge_rbf(xyz, edge_index)
        edge_ori = self._edge_orientations(CaX, edge_index)
        geo_edge_feat = torch.cat([edge_dir, edge_ori, edge_rbf], dim=-1)  # [B, N, K, 18 + 7 + 6 * rbf_num]

        node = self.norm_node(self.node_emb(geo_node_feat))
        edge = self.norm_edge(self.edge_emb(geo_edge_feat))
        return node, edge, edge_index  # [B, N, hidden], [B, N, K, hidden], [B, N, K]

    def _distance(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        _, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False)
        return E_idx

    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))

        def _R(i, j): return R[:, :, :, i, j]

        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _edge_rbf(self, X, edge_index, D_min=0., D_max=20.):
        D_count = self.rbf_num
        K = edge_index.shape[-1]
        X_expand = X.unsqueeze(2).expand(-1, -1, K, -1, -1)
        X_neigh = X_expand.gather(1, edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 6, 3))
        CaX = X[:, :, 1]
        D = torch.norm(X_neigh - CaX.unsqueeze(2).unsqueeze(3), dim=-1)
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1, 1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1).to(D.device)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        RBF = RBF.flatten(-2)
        return RBF

    def _node_rbf(self, X, D_min=0., D_max=20.):
        D_count = self.rbf_num
        D_mu = torch.linspace(D_min, D_max, D_count, device=X.device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        rel_list = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4], [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]]
        D = torch.norm(X[:, :, rel_list[0]] - X[:, :, rel_list[1]], dim=-1)
        D_expand = torch.unsqueeze(D, -1)
        D_out = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2).flatten(-2)
        return D_out

    def _node_angle(self, X, mask, eps=1e-7):
        B = X.shape[0]
        X = torch.reshape(X[:, :, :3], [B, 3 * X.shape[1], 3])
        dX = X[:, 1:] - X[:, :-1]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2]
        u_1 = U[:, 1:-1]
        u_0 = U[:, 2:]
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [B, -1, 3])
        dihedral = torch.cat([torch.cos(D), torch.sin(D)], -1)
        cosD = (u_2 * u_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.acos(cosD)
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [B, -1, 3])
        bond_angles = torch.cat((torch.cos(D), torch.sin(D)), -1)
        node_angles = torch.cat((dihedral, bond_angles), -1)
        for idx in range(node_angles.shape[0]):
            node_angles[idx][mask[idx].sum() - 1] = 0
        return node_angles

    def _node_direct(self, X, edge_index):
        A_n = X[:, :, 0]
        A_ca = X[:, :, 1]
        A_c = X[:, :, 2]
        u = F.normalize(A_n - A_ca, dim=-1)
        v = F.normalize(A_ca - A_c, dim=-1)
        b = F.normalize(u - v, dim=-1)
        n = F.normalize(torch.cross(u, v, dim=-1), dim=-1)
        local_frame = torch.stack([b, n, torch.cross(b, n, dim=-1)], dim=-1)
        t = F.normalize(X[:, :, [0, 2, 3, 4, 5]] - A_ca.unsqueeze(-2), dim=-1)
        node_direct = torch.matmul(t, local_frame).flatten(-2)
        X_expand = X.unsqueeze(2).expand(-1, -1, self.top_k, -1, -1)
        X_neigh = X_expand.gather(1, edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 6, 3))
        t = F.normalize(X_neigh - A_ca.unsqueeze(-2).unsqueeze(-2), dim=-1)
        edge_direction = torch.matmul(t, local_frame.unsqueeze(2)).flatten(-2)
        return node_direct, edge_direction

    def _edge_orientations(self, X, E_idx, eps=1e-6):
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2, dim=-1)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)
        O_neighbors = Func.gather_nodes(O, E_idx)
        X_neighbors = Func.gather_nodes(X, E_idx)
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)
        O_features = torch.cat((dU, Q), dim=-1)
        return O_features


class MDBind(nn.Module):
    def __init__(self, rfeat_dim=1024, ligand_dim=64, hidden_dim=256, heads=4, augment_eps=0.1, rbf_num=16, top_k=5,
                 attn_drop=0.2, dropout=0.2, num_layers=2):
        super(MDBind, self).__init__()
        self.augment_eps = augment_eps

        self.in_mlp = LMlp(rfeat_dim, rfeat_dim // 2, hidden_dim)
        self.lig_mlp = easyMLP(ligand_dim, hidden_dim)
        self.edge_feature = EdgeFeature(hidden_dim, rbf_num, top_k)

        self.f_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim, eps=1e-6)
        )

        self.conv_layers = nn.ModuleList([
            InterTransformer(hidden_dim, hidden_dim * 2, heads, attn_drop, dropout)
            for _ in range(num_layers)])

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2)  # 2 分类：非结合(0) 与 结合(1)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask):
        if self.training and self.augment_eps > 0.:
            xyz = xyz + torch.randn_like(xyz) * self.augment_eps
            rfeat = rfeat + torch.randn_like(rfeat) * self.augment_eps
            ligand = ligand + torch.randn_like(ligand) * self.augment_eps

        rfeat = self.in_mlp(rfeat)
        ligand = self.lig_mlp(ligand)

        node, edge, e_idx = self.edge_feature(xyz, mask, angles)

        node = torch.cat([rfeat, node], dim=-1)
        node = self.f_mlp(node)  # [B, N, hidden_dim]

        mask_attend = Func.gather_nodes(mask.unsqueeze(-1), e_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.conv_layers:
            node, edge, ligand = layer(node, edge, e_idx, cmaps, ligand, lig_node_paths, lig_mask, mask, mask_attend)

        out = self.out_mlp(node)  # [B, N, 2]
        evidence = F.softplus(out)  # 确保 evidence > 0
        return evidence