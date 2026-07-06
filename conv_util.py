import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import func_help as Func
import numpy as np
# from config import *
device = torch.device('cuda')

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List

class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        super().__init__()
        if max_path_distance < 1:
            raise ValueError("max_path_distance must be >= 1")
        self.max_path_distance = max_path_distance
        # indices: 0..max_path_distance, where 0 is padding/no-path
        self.embedding = nn.Embedding(num_embeddings=max_path_distance + 1,
                                      embedding_dim=1,
                                      padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, node_paths_length: torch.LongTensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param node_paths_length: LongTensor [B, L, L], values in {0,1,...}
        :param node_mask: Optional BoolTensor [B, L] (True for valid nodes)
        :return: FloatTensor [B, L, L] spatial bias
        """
        # device = next(self.parameters()).device
        # node_paths_length = node_paths_length.to(device)
        indices = torch.clamp(node_paths_length, min=0, max=self.max_path_distance).long()
        emb = self.embedding(indices).squeeze(-1)  # [B, L, L]
        if node_mask is not None:
            mask = node_mask.to(device).bool()
            pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # [B, L, L]
            emb = emb * pair_mask.to(emb.dtype)
        return emb


class GraphormerLayer(nn.Module):
    """
    Single Graphormer layer (multi-head attention + FFN) for batched ligand graphs.
    Inputs:
      - batch_ligand: [B, L, D]
      - batch_lig_mask: [B, L] bool
      - batch_lig_node_paths: [B, L, L] long (0 = no-path/padding)
    Output:
      - [B, L, D] (padding nodes zeroed)
    """
    def __init__(self,
                 node_dim: int,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 max_path_distance: int = 10,
                 dropout: float = 0.0,
                 use_spatial: bool = True):
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError("node_dim must be divisible by num_heads")
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.use_spatial = use_spatial

        # projections
        self.q_lin = nn.Linear(node_dim, node_dim)
        self.k_lin = nn.Linear(node_dim, node_dim)
        self.v_lin = nn.Linear(node_dim, node_dim)
        self.out_lin = nn.Linear(node_dim, node_dim)

        # spatial encoding per layer (can be replaced by shared encoder in GraphormerEncoder)
        self.spatial_enc = SpatialEncoding(max_path_distance=max_path_distance) if use_spatial else None

        # norms and FFN
        self.ln1 = nn.LayerNorm(node_dim)
        # self.ln2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, node_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                batch_ligand: torch.Tensor,
                batch_lig_node_paths: torch.LongTensor,
                batch_lig_mask: torch.Tensor,
                spatial_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param batch_ligand: [B, L, D]
        :param batch_lig_mask: [B, L] bool
        :param batch_lig_node_paths: [B, L, L] long
        :param spatial_bias: Optional precomputed [B, L, L] float to use instead of layer's own spatial_enc
        :return: [B, L, D]
        """
        device = batch_ligand.device
        B, L, D = batch_ligand.shape
        assert D == self.node_dim

        # x_norm = self.ln1(batch_ligand)  # [B, L, D]

        # projections and reshape to [B, H, L, head_dim]
        q = self.q_lin(batch_ligand).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_lin(batch_ligand).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_lin(batch_ligand).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, L, L]

        # spatial bias
        if spatial_bias is None and self.use_spatial:
            b = self.spatial_enc(batch_lig_node_paths, node_mask=batch_lig_mask)  # [B, L, L]
        elif spatial_bias is not None:
            b = spatial_bias
        else:
            b = torch.zeros((B, L, L), device=device, dtype=logits.dtype)
        logits = logits + b.unsqueeze(1)  # broadcast to [B, H, L, L]

        # mask invalid pairs
        pair_mask = (batch_lig_mask.unsqueeze(-1) & batch_lig_mask.unsqueeze(-2)).to(device)  # [B, L, L]
        neg_inf = -1e9
        mask_float = pair_mask.unsqueeze(1).to(torch.bool)  # [B,1,L,L]
        logits = logits.masked_fill(~mask_float, neg_inf)

        attn = torch.softmax(logits, dim=-1)  # [B, H, L, L]
        attn = self.dropout(attn)

        out_heads = torch.matmul(attn, v)  # [B, H, L, head_dim]
        out = out_heads.permute(0, 2, 1, 3).contiguous().view(B, L, D)  # [B, L, D]
        out = self.out_lin(out)

        # residual + dropout
        x = batch_ligand + self.dropout(out)

        # FFN
        x = x + self.ff(self.ln1(x))

        # zero padding nodes
        node_mask_f = batch_lig_mask.to(device).unsqueeze(-1).to(x.dtype)
        x = x * node_mask_f

        return x

#
# class GraphormerEncoder(nn.Module):
#     """
#     Stack multiple GraphormerLayer to form an encoder.
#     - n_layers: number of layers
#     - share_spatial: if True, use one shared BatchedSpatialEncoding for all layers (memory efficient)
#                      if False, each layer has its own spatial encoder
#     """
#     def __init__(self,
#                  n_layers: int,
#                  node_dim: int,
#                  num_heads: int = 8,
#                  ff_dim: int = 2048,
#                  max_path_distance: int = 10,
#                  dropout: float = 0.0,
#                  share_spatial: bool = False):
#         super().__init__()
#         self.n_layers = n_layers
#         self.layers: nn.ModuleList = nn.ModuleList()
#         self.share_spatial = share_spatial
#
#         # optional shared spatial encoder
#         self.shared_spatial = SpatialEncoding(max_path_distance=max_path_distance) if share_spatial else None
#
#         for i in range(n_layers):
#             # if sharing spatial, pass use_spatial=False to layer to avoid duplicate embeddings
#             use_spatial = not share_spatial
#             layer = GraphormerLayer(node_dim=node_dim,
#                                     num_heads=num_heads,
#                                     ff_dim=ff_dim,
#                                     max_path_distance=max_path_distance,
#                                     dropout=dropout,
#                                     use_spatial=use_spatial)
#             self.layers.append(layer)
#
#     def forward(self,
#                 batch_ligand: torch.Tensor,
#                 batch_lig_mask: torch.Tensor,
#                 batch_lig_node_paths: torch.LongTensor) -> torch.Tensor:
#         """
#         :param batch_ligand: [B, L, D]
#         :param batch_lig_mask: [B, L] bool
#         :param batch_lig_node_paths: [B, L, L] long
#         :return: [B, L, D]
#         """
#         x = batch_ligand
#         spatial_bias = None
#         if self.shared_spatial is not None:
#             spatial_bias = self.shared_spatial(batch_lig_node_paths, node_mask=batch_lig_mask)  # [B, L, L]
#
#         for layer in self.layers:
#             x = layer(x, batch_lig_mask, batch_lig_node_paths, spatial_bias=spatial_bias)
#
#         return x

class CrossAttention(nn.Module):
    def __init__(self, d_model=256, heads=8, attn_dropout=0.1, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.query = nn.Linear(d_model, d_model * heads)
        self.key = nn.Linear(d_model, d_model * heads)
        self.value = nn.Linear(d_model, d_model * heads)
        self.out = nn.Linear(d_model * heads, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, node_mask, ligand, ligand_mask):
        """
        node: [B, N, C]
        node_mask: [B, N] (bool)
        ligand: [B, L, C]
        ligand_mask: [B, L] (bool)
        """
        B, N, C = node.size()
        _, L, _ = ligand.size()

        # project to multi-head
        query = self.query(node).view(B, N, self.heads, self.d_model).transpose(1, 2)   # [B, heads, N, C]
        key   = self.key(ligand).view(B, L, self.heads, self.d_model).transpose(1, 2)  # [B, heads, L, C]
        value = self.value(ligand).view(B, L, self.heads, self.d_model).transpose(1, 2) # [B, heads, L, C]

        # attention scores
        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)  # [B, heads, N, L]

        # apply ligand mask: mask invalid ligand positions
        # ligand_mask: [B, L] -> [B, 1, 1, L]
        ligand_mask_exp = ligand_mask.unsqueeze(1).unsqueeze(2)  # broadcast
        # 如果不是布尔类型，转换为布尔
        if ligand_mask_exp.dtype != torch.bool:
            ligand_mask_exp = ligand_mask_exp.bool()
        attn = attn.masked_fill(~ligand_mask_exp, float('-inf'))

        # softmax over ligand dimension
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # weighted sum
        x = torch.matmul(attn, value)  # [B, heads, N, C]

        # apply node mask: zero out invalid node positions
        x = x.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, C*heads]
        x = self.dropout(x)
        x = self.out(x)

        # mask invalid nodes
        x = x * node_mask.unsqueeze(-1)  # [B, N, C]

        return x
   
class LMlp(nn.Module):
    def __init__(self,in_dim,hidden_dim=256,out_dim=256):
        super(LMlp,self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(out_dim)
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = self.ln1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.ln2(x)
        return x
class FeedForward(nn.Module):
    '''
        这是一个前馈神经网络，用于序列数据的处理
    '''
    def __init__(self,d_model=256,d_ff=512,dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        # x: [batch_size,seq_len,d_model]
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4,attn_drop=0.2):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.attn_drop = nn.Dropout(attn_drop)
        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

        self.cmap_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, num_heads)
        )

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(attend_logits.device))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, edge_index, cmap=None, mask_attend=None):

        # Queries, Keys, Values
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)

        # 2. [新增] 提取局部接触图并融合为 Attention Bias
        if cmap is not None:
            # 确保 cmap 维度和设备匹配
            # 将 edge_index 扩展为 [B, N, K, 2] 以便进行特征聚合
            edge_idx_expand = edge_index.unsqueeze(-1).expand(-1, -1, -1, 2)

            # 使用 torch.gather 沿着邻居维度(dim=2)从 [B, N, N, 2] 提取 [B, N, K, 2]
            local_cmap = torch.gather(cmap, dim=2, index=edge_idx_expand)

            # 经过 MLP 映射到多头维度: [B, N, K, 2] -> [B, N, K, heads]
            cmap_bias = self.cmap_proj(local_cmap)

            # 转置以匹配 attend_logits 的形状: [B, N, K, heads] -> [B, N, heads, K]
            cmap_bias = cmap_bias.transpose(-2, -1)

            # 直接将偏置加到 logits 上
            attend_logits = attend_logits + cmap_bias

        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(2).expand(-1,-1,n_heads,-1)
            attend = self._masked_softmax(attend_logits, mask) # [B, L, heads, K]
        else:
            attend = F.softmax(attend_logits, -1)
        attend = self.attn_drop(attend)
        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2,3)) # [B, L, heads, 1, K] × [B, L, heads, K, d]
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update

class InterTransformer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, attn_drop=0.2, dropout=0.2):
        super(InterTransformer, self).__init__()

        self.ProtAttn = NeighborAttention(num_hidden, num_in, num_heads, attn_drop)  # 邻居注意力 关注邻居节点
        self.LigAttn = GraphormerLayer(node_dim=num_hidden, num_heads=num_heads, ff_dim=num_hidden, max_path_distance=15, dropout=dropout, use_spatial=True)
        self.dropout = nn.Dropout(dropout) # dropout layer
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(5)]) # attention后的norm层
        
        self.ProtCrossAttn = CrossAttention(num_hidden, num_heads, attn_drop, dropout) # 与配体的交叉注意力
        self.LigCrossAttn = CrossAttention(num_hidden, num_heads, attn_drop, dropout)


        
        self.prot_dense = FeedForward(num_hidden, num_hidden * 4, dropout) # 前馈神经网络
        self.lig_dense = FeedForward(num_hidden, num_hidden * 4, dropout)  # 前馈神经网络
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden, dropout)
        
    def forward(self, node, edge, edge_index, cmap, ligand, lig_node_paths,lig_mask, mask=None, mask_attend=None):    # mask_attend [B, L, K]
        # node: [B, L, D] edge: [B, L, K, D] E_idx: [B, L, K]
        # Concatenate node_i to h_E_ij       
        """ Parallel computation of full transformer layer """
        # Self-attention  
        h_EV = Func.cat_neighbors_nodes(node, edge, edge_index) # 聚合邻居节点和边特征 [edge K nodej]
         
        dh = self.ProtAttn(node, h_EV, edge_index, cmap=cmap, mask_attend=mask_attend) # [nodei] 同  [edge K*nodej] 进行注意力聚合
        node = self.norm[0](node + self.dropout(dh)) # Add & Norm
        
        dh = self.ProtCrossAttn(node,mask, ligand,lig_mask)
        node = self.norm[1](node + self.dropout(dh)) # Add & Norm
        
        dh = self.prot_dense(node) # 前馈神经网络
        node = self.norm[2](node + self.dropout(dh)) # Add & Norm
        if mask is not None: 
            node = mask.unsqueeze(-1) * node # mask
        # global update for ligand

        ligand = self.LigAttn(ligand, lig_node_paths, lig_mask)
        dh = self.LigCrossAttn(ligand, lig_mask, node, mask)
        ligand = self.norm[3](ligand + self.dropout(dh))  # Add & Norm

        dh = self.lig_dense(ligand)  # 前馈神经网络
        ligand = self.norm[4](ligand + self.dropout(dh))  # Add & Norm
        if lig_mask is not None:
            ligand = lig_mask.unsqueeze(-1) * ligand # mask

        edge = self.edge_update(node, edge, edge_index)
        return node, edge, ligand
        
    
class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_hidden)
        self.EdgeMLP = nn.Sequential(
            nn.Linear(3 * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )

    def forward(self, node, edge, edge_index):
        h_VE = Func.gather_edges(edge, node, edge_index) # [B, N, K, 3*C]
        edge = self.norm(edge + self.dropout(self.EdgeMLP(h_VE)))
        return edge


class Context(nn.Module):
    def __init__(self, num_hidden,dropout=0.2):
        super(Context, self).__init__()
        self.ContextMLP = nn.Sequential(
            nn.Linear(3 * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_hidden)
    def forward(self, node, ligand, mask=None):
        # node [B,N,C] ligand [B,1,C] mask [B,N]
        mean_node = node * mask.unsqueeze(-1)
        mean_node = torch.sum(mean_node,dim=1)/torch.sum(mask,dim=1).unsqueeze(-1)
        max_node,_= torch.max(node,dim=1)
        mean_node = mean_node.unsqueeze(1)
        max_node = max_node.unsqueeze(1)
        h_L = torch.cat([mean_node,max_node,ligand],dim=-1)
        ligand = self.norm(ligand + self.dropout(self.ContextMLP(h_L)))
        return ligand
    
class easyMLP(nn.Module):
    def __init__(self,in_dim=64,out_dim=64):
        super(easyMLP,self).__init__()
        self.fc1 = nn.Sequential(
                nn.LayerNorm(in_dim, eps=1e-6)
                ,nn.Linear(in_dim, out_dim)
                ,nn.LeakyReLU()
                )
    def forward(self,x):
        return self.fc1(x)
