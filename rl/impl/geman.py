"""
impl/geman.py
-----------------------
GEMAN Actor-Critic Network for VRPBTW (Graph-Enhanced Multi-relational Attention Network).

Primary implementation for VRPBTW policy. The network receives a fully pre-processed obs dict from
VRPBTWProblem.state_to_obs — all arrays are already normalised and the
static compatibility graph is already built. No preprocessing happens inside this file.

Obs dict keys (all numpy arrays, produced by state_to_obs):
  node_features     (B, N+1, 5)
  vehicle_features  (B, 2K,  5)
  truck_edge_index  (B, 2, E)   or  (2, E)  — shared across batch
  truck_edge_attr   (B, E, 2)   or  (E, 2)
  drone_edge_index  (B, 2, E)   or  (2, E)
  drone_edge_attr   (B, E, 2)   or  (E, 2)

Architecture
============
Encoder
  NodeEncoder      heterogeneous self-attention (linehaul ↔ backhaul)
                   → Z_node (B, N+1, D),  g_node (B, D)
  VehicleEncoder   self-attention + cross-attention to Z_node
                   → Z_veh  (B, 2K,  D),  g_veh  (B, D)
  GraphEncoder     3-layer multi-relational GNN (truck / drone relations)
                   → Z_graph (B, N+1, D),  g_graph (B, D)

Decoder  (bilevel, two independent heads)
  NodeDecoder      query = f(g_node, mean(Z_veh), g_graph)
                   dot-product attention + tanh clip over Z_node
  VehicleDecoder   query = f(g_veh, mean(Z_node), g_graph)
                   dot-product attention + tanh clip over Z_veh
  Sampling         rejection sampling (up to MAX_REJECTION_TRIES),
                   fallback to masked-flat argmax

Value head  MLP(g_node ‖ g_veh ‖ g_graph)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import globals
from core.network import ActorCritic, _MHA, _FF, _make_norm
from impl.vrpbtw import NODE_FEAT_DIM, VEH_FEAT_DIM

GRAPH_EDGE_DIM = 2  # [cost, time]
GRAPH_NODE_DIM = 4  # [x, y, tw_open, tw_close] — demand excluded; GraphEncoder
# captures routing structure, not delivery semantics


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# NodeEncoder
# ---------------------------------------------------------------------------


class _NodeEncoderLayer(nn.Module):
    def __init__(self, D: int, H: int, dropout: float, use_in: bool):
        super().__init__()
        self.sa = _MHA(D, H, dropout)
        self.ff = _FF(D, dropout)
        self.norm1 = _make_norm(use_in, D)
        self.norm2 = _make_norm(use_in, D)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm1(h + self.sa(h, h, h))
        h = torch.nan_to_num(h, nan=0.0)
        h = self.norm2(h + self.ff(h))
        h = torch.nan_to_num(h, nan=0.0)
        return h


class NodeEncoder(nn.Module):
    def __init__(
        self,
        D: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_instance_norm: bool,
    ):
        super().__init__()
        self.input_proj = nn.Linear(NODE_FEAT_DIM, D)
        self.layers = nn.ModuleList(
            [
                _NodeEncoderLayer(D, n_heads, dropout, use_instance_norm)
                for _ in range(n_layers)
            ]
        )
        self.pool = nn.Linear(D, 1)

    def forward(self, node_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feat)
        for layer in self.layers:
            h = layer(h)
        w = torch.softmax(self.pool(h), dim=1)  # (B, N+1, 1)
        return h, (w * h).sum(dim=1)            # (B, N+1, D), (B, D)


# ---------------------------------------------------------------------------
# VehicleEncoder
# ---------------------------------------------------------------------------


class _VehicleEncoderLayer(nn.Module):
    def __init__(self, D: int, H: int, dropout: float, use_in: bool):
        super().__init__()
        self.sa = _MHA(D, H, dropout)
        self.ff = _FF(D, dropout)
        self.norm1 = _make_norm(use_in, D)
        self.norm2 = _make_norm(use_in, D)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm1(h + self.sa(h, h, h))
        h = torch.nan_to_num(h, nan=0.0)
        h = self.norm2(h + self.ff(h))
        h = torch.nan_to_num(h, nan=0.0)
        return h


class VehicleEncoder(nn.Module):
    def __init__(
        self,
        D: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_instance_norm: bool,
    ):
        super().__init__()
        self.input_proj = nn.Linear(VEH_FEAT_DIM, D)
        self.layers = nn.ModuleList(
            [
                _VehicleEncoderLayer(D, n_heads, dropout, use_instance_norm)
                for _ in range(n_layers)
            ]
        )
        self.pool = nn.Linear(D, 1)

    def forward(self, veh_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(veh_feat)
        for layer in self.layers:
            h = layer(h)
        w = torch.softmax(self.pool(h), dim=1)  # (B, 2K, 1)
        return h, (w * h).sum(dim=1)            # (B, 2K, D), (B, D)


# ---------------------------------------------------------------------------
# GraphEncoder — multi-relational GNN
# ---------------------------------------------------------------------------


class _MRGNNLayer(nn.Module):
    """
    One multi-relational GNN layer (truck + drone relations).

    For each relation:
      msg(i→j) = MLP( h_i ‖ edge_feat_{ij} )
      agg(j)   = mean over incoming messages

    Update:
      h' = LayerNorm( h + proj( cat[agg_truck, agg_drone] ) )
    """

    def __init__(self, D: int, edge_feat_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_truck = _mlp(D + edge_feat_dim, D * 2, D, dropout)
        self.msg_drone = _mlp(D + edge_feat_dim, D * 2, D, dropout)
        self.update = nn.Linear(D * 2, D)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def _pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mlp: nn.Module,
    ) -> torch.Tensor:
        """
        h          : (B, N, D)
        edge_index : (2, E)
        edge_attr  : (E, ef)   or (B, E, ef)
        returns      (B, N, D)  mean-aggregated messages
        """
        B, N, D = h.shape
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        h_src = h[:, src, :]  # (B, E, D)
        if edge_attr.dim() == 2:
            ea = edge_attr.unsqueeze(0).expand(B, -1, -1)  # (B, E, ef)
        else:
            ea = edge_attr
        msg = mlp(torch.cat([h_src, ea], dim=-1))  # (B, E, D)

        agg = torch.zeros(B, N, D, device=h.device, dtype=h.dtype)
        count = torch.zeros(N, device=h.device, dtype=h.dtype)
        dst_e = dst.unsqueeze(0).unsqueeze(-1).expand(B, E, D)
        agg.scatter_add_(1, dst_e, msg)
        count.scatter_add_(0, dst, torch.ones(E, device=h.device))
        count = count.clamp(min=1.0).view(1, N, 1)
        return agg / count

    def forward(
        self,
        h: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea: torch.Tensor,
    ) -> torch.Tensor:
        agg_t = self._pass(h, t_ei, t_ea, self.msg_truck)
        agg_d = self._pass(h, d_ei, d_ea, self.msg_drone)
        upd = self.drop(self.update(torch.cat([agg_t, agg_d], dim=-1)))
        return self.norm(h + upd)


class GraphEncoder(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        D = cfg.get("embedding_dim", 128)
        n_layers = cfg.get("n_layers", 3)
        dropout = cfg.get("dropout", 0.1)
        use_instance_norm = cfg.get("use_instance_norm", False)

        self.input_proj = nn.Linear(GRAPH_NODE_DIM, D)
        self.layers = nn.ModuleList(
            [_MRGNNLayer(D, GRAPH_EDGE_DIM, dropout) for _ in range(n_layers)]
        )
        self.out_norm = _make_norm(use_instance_norm, D)
        self.pool = nn.Linear(D, 1)

    def forward(
        self,
        node_feat: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feat)
        for layer in self.layers:
            h = layer(h, t_ei, t_ea, d_ei, d_ea)
        h = self.out_norm(h)
        w = torch.softmax(self.pool(h), dim=1)  # (B, N+1, 1)
        return h, (w * h).sum(dim=1)            # (B, N+1, D), (B, D)


# ---------------------------------------------------------------------------
# EGALayer - Expander Graph Attention (for ablation studies)
# ---------------------------------------------------------------------------


class _EGALayer(nn.Module):
    """Single EGA layer with sparse attention via dynamic graph construction.

    Uses k-NN + hierarchical + depot connections for sparse attention.
    Includes relative positional encoding based on euclidean distances.
    """

    def __init__(self, D: int, dropout: float = 0.0, d_sparse: int = 16):
        super().__init__()
        self.D = D
        self.dropout = nn.Dropout(dropout)

        # Multi-head attention projections
        self.n_heads = 4  # Default, matches GEMAN
        self.d_k = D // self.n_heads
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)

        # Relative positional encoding (distance-based)
        self.d_sparse = d_sparse
        self.rel_pos_embed = nn.Embedding(32, self.n_heads * self.d_k)  # 32 distance buckets

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.ReLU(),
            nn.Linear(D * 2, D),
        )

        # Normalization
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    @staticmethod
    def _build_sparse_mask(B: int, N: int, device: torch.device, k: int = 3) -> torch.Tensor:
        """Build sparse attention mask using fixed k-NN pattern.

        Returns:
            mask: (B, N, N) with 0 where connected, -inf where not
        """
        # For simplicity: allow each node to attend to itself, neighbors, and depot
        # Can be extended with dynamic k-NN based on embeddings
        adj = torch.zeros(B, N, N, device=device)

        # Self-attention
        adj[:, range(N), range(N)] = 1

        # Depot (0) connects to all
        adj[:, 0, :] = 1
        adj[:, :, 0] = 1

        # k-NN approximation: sequential neighbors (spatial locality)
        for i in range(N):
            for j in range(max(0, i - k), min(N, i + k + 1)):
                adj[:, i, j] = 1
                adj[:, j, i] = 1

        # Convert to attention mask: 0 → 0, 1 → 0, else → -inf
        mask = torch.where(adj > 0, torch.tensor(0.0, device=device), torch.tensor(float('-inf'), device=device))
        return mask

    def _compute_rel_pos_encoding(self, node_coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding based on euclidean distances.

        Args:
            node_coords: (B, N, 2) node coordinates
            mask: (B, N, N) attention mask

        Returns:
            r_ij: (B, N, N, n_heads, d_k) relative position embeddings
        """
        B, N, _ = node_coords.shape
        device = node_coords.device

        # Pairwise distances
        coords_i = node_coords.unsqueeze(2)  # (B, N, 1, 2)
        coords_j = node_coords.unsqueeze(1)  # (B, 1, N, 2)
        distances = torch.norm(coords_i - coords_j, dim=-1)  # (B, N, N)

        # Bucket distances into 32 bins
        max_dist = distances.max()
        dist_buckets = (distances / (max_dist + 1e-6) * 31).long().clamp(0, 31)

        # Look up embeddings
        r_ij = self.rel_pos_embed(dist_buckets)  # (B, N, N, n_heads * d_k)
        r_ij = r_ij.view(B, N, N, self.n_heads, self.d_k)

        return r_ij

    def forward(self, h: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """EGA forward pass with sparse attention.

        Args:
            h: (B, N, D) node embeddings
            coords: (B, N, 2) node coordinates for distance-based relative positions

        Returns:
            h: (B, N, D) updated node embeddings
        """
        B, N, D = h.shape
        device = h.device

        # Build sparse attention mask
        mask = self._build_sparse_mask(B, N, device, k=3)  # 3-NN sparse attention

        # Compute Q, K, V
        Q = self.W_Q(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, N, d_k)
        K = self.W_K(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, N, d_k)
        V = self.W_V(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, N, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, n_heads, N, N)

        # Add relative position encoding if coordinates provided
        if coords is not None:
            r_ij = self._compute_rel_pos_encoding(coords, mask)  # (B, N, N, n_heads, d_k)
            r_ij = r_ij.permute(0, 3, 1, 2, 4)  # (B, n_heads, N, N, d_k)
            Q_expanded = Q.unsqueeze(-2)  # (B, n_heads, N, 1, d_k)
            rel_scores = (Q_expanded * r_ij).sum(dim=-1)  # (B, n_heads, N, N)
            scores = scores + rel_scores

        # Apply sparse mask
        scores = scores + mask.unsqueeze(1)  # (B, 1, N, N)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        h_attn = torch.matmul(attn_weights, V)  # (B, n_heads, N, d_k)
        h_attn = h_attn.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        h_attn = self.W_O(h_attn)

        # Residual + norm
        h = self.norm1(h + h_attn)

        # FFN + residual + norm
        h = self.norm2(h + self.ffn(h))

        return h


class EGAEncoder(nn.Module):
    """Expander Graph Attention Encoder for VRPBTW.

    Alternative to GraphEncoder using sparse attention with dynamic graph construction.
    Useful for ablation studies to compare attention mechanisms.

    Compatible with GraphEncoder interface:
        forward(node_feat, t_ei, t_ea, d_ei, d_ea) → (Z_graph, g_graph)
    """

    def __init__(self, D: int, n_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(GRAPH_NODE_DIM, D)
        self.layers = nn.ModuleList([_EGALayer(D, dropout, d_sparse=16) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(D)
        self.pool = nn.Linear(D, 1)

    def forward(
        self,
        node_feat: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_feat: (B, N+1, GRAPH_NODE_DIM) node features
            t_ei, t_ea, d_ei, d_ea: Edge indices/attributes (unused, for interface compatibility)
            coords: (B, N+1, 2) node coordinates for relative position encoding

        Returns:
            Z: (B, N+1, D) node embeddings
            g: (B, D) pooled global embedding
        """
        h = self.input_proj(node_feat)

        for layer in self.layers:
            h = layer(h, coords)

        h = self.norm(h)
        w = torch.softmax(self.pool(h), dim=1)  # (B, N+1, 1)
        return h, (w * h).sum(dim=1)  # (B, N+1, D), (B, D)


# ---------------------------------------------------------------------------
# MEGAGraphEncoder Variants - Multi-Edge Graph Aggregation (for ablation studies)
# ---------------------------------------------------------------------------
# Three implementations: MLP (simplest), GCN (normalized), GAT (attention-based)


class MLP_MEGAGraphEncoder(nn.Module):
    """MLP-based MEGA encoder: two-stage aggregation with MLP message passing.

    Stage 1: Project edge attributes [cost, time] → D-dim embeddings
    Stage 2: MLP([h_src || e_proj]) → mean aggregation

    Simplest variant, fastest, good for sparse graphs.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        D = cfg.get("embedding_dim", 128)
        n_layers = cfg.get("n_layers", 3)
        dropout = cfg.get("dropout", 0.1)
        use_instance_norm = cfg.get("use_instance_norm", False)

        self.D = D
        self.input_proj = nn.Linear(GRAPH_NODE_DIM, D)
        self.edge_truck_proj = nn.Linear(GRAPH_EDGE_DIM, D)
        self.edge_drone_proj = nn.Linear(GRAPH_EDGE_DIM, D)
        self.layers = nn.ModuleList(
            [_MLPMEGAMRGNNLayer(D, dropout) for _ in range(n_layers)]
        )
        self.out_norm = _make_norm(use_instance_norm, D)
        self.pool = nn.Linear(D, 1)

    def forward(
        self,
        node_feat: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feat)
        t_ea_proj = self.edge_truck_proj(t_ea)
        d_ea_proj = self.edge_drone_proj(d_ea)
        for layer in self.layers:
            h = layer(h, t_ei, t_ea_proj, d_ei, d_ea_proj)
        h = self.out_norm(h)
        w = torch.softmax(self.pool(h), dim=1)
        return h, (w * h).sum(dim=1)


class _MLPMEGAMRGNNLayer(nn.Module):
    """MLP message passing with mean aggregation."""

    def __init__(self, D: int, dropout: float = 0.0):
        super().__init__()
        self.msg_truck = _mlp(D + D, D * 2, D, dropout)
        self.msg_drone = _mlp(D + D, D * 2, D, dropout)
        self.update = nn.Linear(D * 2, D)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def _pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_proj: torch.Tensor,
        mlp: nn.Module,
    ) -> torch.Tensor:
        B, N, D = h.shape
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        h_src = h[:, src, :]
        if edge_attr_proj.dim() == 2:
            ea_proj = edge_attr_proj.unsqueeze(0).expand(B, -1, -1)
        else:
            ea_proj = edge_attr_proj

        msg = mlp(torch.cat([h_src, ea_proj], dim=-1))

        agg = torch.zeros(B, N, D, device=h.device, dtype=h.dtype)
        count = torch.zeros(N, device=h.device, dtype=h.dtype)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, E, D)
        agg.scatter_add_(1, dst_expanded, msg)
        count.scatter_add_(0, dst, torch.ones(E, device=h.device))
        count = count.clamp(min=1.0).view(1, N, 1)
        return agg / count

    def forward(
        self,
        h: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea_proj: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea_proj: torch.Tensor,
    ) -> torch.Tensor:
        agg_t = self._pass(h, t_ei, t_ea_proj, self.msg_truck)
        agg_d = self._pass(h, d_ei, d_ea_proj, self.msg_drone)
        upd = self.drop(self.update(torch.cat([agg_t, agg_d], dim=-1)))
        return self.norm(h + upd)


class GCN_MEGAGraphEncoder(nn.Module):
    """GCN-based MEGA encoder: symmetric degree normalization D^(-1/2) A D^(-1/2).

    Stage 1: Project edge attributes [cost, time] → D-dim embeddings
    Stage 2: MLP([h_src || e_proj]) → degree-normalized aggregation

    Better for graphs with skewed degree distribution.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        D = cfg.get("embedding_dim", 128)
        n_layers = cfg.get("n_layers", 3)
        dropout = cfg.get("dropout", 0.1)
        use_instance_norm = cfg.get("use_instance_norm", False)

        self.D = D
        self.input_proj = nn.Linear(GRAPH_NODE_DIM, D)
        self.edge_truck_proj = nn.Linear(GRAPH_EDGE_DIM, D)
        self.edge_drone_proj = nn.Linear(GRAPH_EDGE_DIM, D)
        self.layers = nn.ModuleList(
            [_GCNMEGAMRGNNLayer(D, dropout) for _ in range(n_layers)]
        )
        self.out_norm = _make_norm(use_instance_norm, D)
        self.pool = nn.Linear(D, 1)

    def forward(
        self,
        node_feat: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feat)
        t_ea_proj = self.edge_truck_proj(t_ea)
        d_ea_proj = self.edge_drone_proj(d_ea)
        for layer in self.layers:
            h = layer(h, t_ei, t_ea_proj, d_ei, d_ea_proj)
        h = self.out_norm(h)
        w = torch.softmax(self.pool(h), dim=1)
        return h, (w * h).sum(dim=1)


class _GCNMEGAMRGNNLayer(nn.Module):
    """GCN-style message passing with symmetric degree normalization."""

    def __init__(self, D: int, dropout: float = 0.0):
        super().__init__()
        self.msg_truck = _mlp(D + D, D * 2, D, dropout)
        self.msg_drone = _mlp(D + D, D * 2, D, dropout)
        self.update = nn.Linear(D * 2, D)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def _pass(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_proj: torch.Tensor,
        mlp: nn.Module,
    ) -> torch.Tensor:
        """GCN aggregation: D^(-1/2) A D^(-1/2) style normalization."""
        B, N, D = h.shape
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        h_src = h[:, src, :]
        if edge_attr_proj.dim() == 2:
            ea_proj = edge_attr_proj.unsqueeze(0).expand(B, -1, -1)
        else:
            ea_proj = edge_attr_proj

        msg = mlp(torch.cat([h_src, ea_proj], dim=-1))

        # Compute in-degree and out-degree for normalization
        in_degree = torch.zeros(N, device=h.device, dtype=h.dtype)
        out_degree = torch.zeros(N, device=h.device, dtype=h.dtype)
        in_degree.scatter_add_(0, dst, torch.ones(E, device=h.device, dtype=h.dtype))
        out_degree.scatter_add_(0, src, torch.ones(E, device=h.device, dtype=h.dtype))

        # Compute D^(-1/2)
        d_inv_sqrt = torch.ones(N, device=h.device, dtype=h.dtype)
        d_inv_sqrt[in_degree > 0] = 1.0 / torch.sqrt(in_degree[in_degree > 0])
        d_inv_sqrt_out = torch.ones(N, device=h.device, dtype=h.dtype)
        d_inv_sqrt_out[out_degree > 0] = 1.0 / torch.sqrt(out_degree[out_degree > 0])

        # Apply symmetric normalization: D^(-1/2) A D^(-1/2)
        norm_coeff = d_inv_sqrt_out[src] * d_inv_sqrt[dst]  # (E,)
        msg = msg * norm_coeff.view(1, E, 1)  # (B, E, D)

        # Scatter-add (already normalized)
        agg = torch.zeros(B, N, D, device=h.device, dtype=h.dtype)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, E, D)
        agg.scatter_add_(1, dst_expanded, msg)

        return agg

    def forward(
        self,
        h: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea_proj: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea_proj: torch.Tensor,
    ) -> torch.Tensor:
        agg_t = self._pass(h, t_ei, t_ea_proj, self.msg_truck)
        agg_d = self._pass(h, d_ei, d_ea_proj, self.msg_drone)
        upd = self.drop(self.update(torch.cat([agg_t, agg_d], dim=-1)))
        return self.norm(h + upd)


class GAT_MEGAGraphEncoder(nn.Module):
    """GAT-based MEGA encoder: multi-head attention over edge messages.

    Stage 1: Project edge attributes [cost, time] → D-dim embeddings
    Stage 2: Multi-head attention([h_src || e_proj]) → attention-weighted aggregation

    Best for learning which edges matter most. Most expressive but slower.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        D = cfg.get("embedding_dim", 128)
        n_layers = cfg.get("n_layers", 3)
        dropout = cfg.get("dropout", 0.1)
        use_instance_norm = cfg.get("use_instance_norm", False)
        n_heads = cfg.get("n_heads", 4)

        self.D = D
        self.input_proj = nn.Linear(GRAPH_NODE_DIM, D)
        self.edge_truck_proj = nn.Linear(GRAPH_EDGE_DIM, D)
        self.edge_drone_proj = nn.Linear(GRAPH_EDGE_DIM, D)
        self.layers = nn.ModuleList(
            [_GATMEGAMRGNNLayer(D, dropout, n_heads) for _ in range(n_layers)]
        )
        self.out_norm = _make_norm(use_instance_norm, D)
        self.pool = nn.Linear(D, 1)

    def forward(
        self,
        node_feat: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_feat)
        t_ea_proj = self.edge_truck_proj(t_ea)
        d_ea_proj = self.edge_drone_proj(d_ea)
        for layer in self.layers:
            h = layer(h, t_ei, t_ea_proj, d_ei, d_ea_proj)
        h = self.out_norm(h)
        w = torch.softmax(self.pool(h), dim=1)
        return h, (w * h).sum(dim=1)


class _GATMEGAMRGNNLayer(nn.Module):
    """GAT-style message passing with learned attention weights."""

    def __init__(self, D: int, dropout: float = 0.0, n_heads: int = 4):
        super().__init__()
        assert D % n_heads == 0, f"D ({D}) must be divisible by n_heads ({n_heads})"
        self.D = D
        self.n_heads = n_heads
        self.d_k = D // n_heads

        # Truck attention
        self.q_truck = nn.Linear(D, D, bias=False)
        self.k_truck = nn.Linear(D, D, bias=False)
        self.v_truck = nn.Linear(D, D, bias=False)
        self.edge_attn_truck = nn.Linear(D, n_heads)
        self.out_truck = nn.Linear(D, D)

        # Drone attention
        self.q_drone = nn.Linear(D, D, bias=False)
        self.k_drone = nn.Linear(D, D, bias=False)
        self.v_drone = nn.Linear(D, D, bias=False)
        self.edge_attn_drone = nn.Linear(D, n_heads)
        self.out_drone = nn.Linear(D, D)

        # Update
        self.update = nn.Linear(D * 2, D)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

    def _pass_attention(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_proj: torch.Tensor,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        edge_attn: nn.Module,
        out_proj: nn.Module,
    ) -> torch.Tensor:
        """Multi-head attention aggregation with edge attribute modulation."""
        B, N, D = h.shape
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        if edge_attr_proj.dim() == 2:
            ea_proj = edge_attr_proj.unsqueeze(0).expand(B, -1, -1)
        else:
            ea_proj = edge_attr_proj

        # Project to Q, K, V and reshape for multi-head
        q = q_proj(h).view(B, N, self.n_heads, self.d_k)  # (B, N, heads, d_k)
        k = k_proj(h).view(B, N, self.n_heads, self.d_k)
        v = v_proj(h).view(B, N, self.n_heads, self.d_k)

        # Extract Q, K, V for edges
        q_src = q[:, src, :, :]  # (B, E, heads, d_k)
        k_dst = k[:, dst, :, :]
        v_src = v[:, src, :, :]

        # Scaled dot-product attention
        scores = torch.matmul(q_src, k_dst.transpose(-2, -1))  # (B, E, heads, d_k, d_k)
        scores = scores / math.sqrt(self.d_k)
        scores = scores.squeeze(-1)  # (B, E, heads, d_k)

        # Modulate attention by edge attributes
        edge_bias = edge_attn(ea_proj)  # (B, E, heads)
        scores = scores.mean(dim=-1) + edge_bias  # (B, E, heads)

        # Softmax per destination node (only over its incoming edges)
        # Build mask: which edges go to which destination
        attn_weights = torch.zeros(B, N, E, self.n_heads, device=h.device)
        for i in range(E):
            attn_weights[:, dst[i], i, :] = scores[:, i, :]

        # Softmax over source edges per destination
        attn_weights = torch.softmax(attn_weights, dim=2)  # (B, N, E, heads)
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention to values and aggregate
        # For each destination, sum weighted source values
        v_expanded = v_src.unsqueeze(1)  # (B, 1, E, heads, d_k)
        weighted_v = attn_weights.unsqueeze(-1) * v_expanded  # (B, N, E, heads, d_k)
        agg = weighted_v.sum(dim=2)  # (B, N, heads, d_k)

        # Reshape back to (B, N, D)
        agg = agg.reshape(B, N, D)
        agg = out_proj(agg)

        return agg

    def forward(
        self,
        h: torch.Tensor,
        t_ei: torch.Tensor,
        t_ea_proj: torch.Tensor,
        d_ei: torch.Tensor,
        d_ea_proj: torch.Tensor,
    ) -> torch.Tensor:
        agg_t = self._pass_attention(
            h, t_ei, t_ea_proj, self.q_truck, self.k_truck, self.v_truck,
            self.edge_attn_truck, self.out_truck,
        )
        agg_d = self._pass_attention(
            h, d_ei, d_ea_proj, self.q_drone, self.k_drone, self.v_drone,
            self.edge_attn_drone, self.out_drone,
        )
        upd = self.drop(self.update(torch.cat([agg_t, agg_d], dim=-1)))
        return self.norm(h + upd)


# ---------------------------------------------------------------------------
# ValueHead - Multi-layer value function approximator
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Configurable multi-layer value head for state value estimation.

    Args:
        input_dim: Input dimension (typically D * 3 from concatenated embeddings)
        hidden_dims: List of hidden layer sizes (e.g., [128, 128])
        dropout: Dropout rate applied after each layer
        use_dropout: Whether to apply dropout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        use_dropout: bool = True,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        if use_dropout:
            layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                ]
            )
            if use_dropout:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate state value.

        Args:
            x: (B, input_dim) tensor

        Returns:
            (B, 1) tensor of value estimates
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# NodeDecoder / VehicleDecoder
# ---------------------------------------------------------------------------


class NodeDecoder(nn.Module):
    def __init__(
        self,
        D: int,
        clip: float = 10.0,
        context_hidden_dim: int = 128,
        context_dropout: float = 0.0,
    ):
        super().__init__()
        self.clip = clip
        layers = [nn.Linear(D * 3, context_hidden_dim), nn.ReLU()]
        if context_dropout > 0:
            layers.append(nn.Dropout(context_dropout))
        self.ctx_proj = nn.Sequential(*layers)
        self.Wq = nn.Linear(context_hidden_dim, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self._scale: Optional[float] = None

    def forward(
        self,
        g_node: torch.Tensor,
        Z_veh: torch.Tensor,
        g_graph: torch.Tensor,
        Z_node: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._scale is None:
            self._scale = math.sqrt(Z_node.shape[-1])
        z_veh_mean = Z_veh.mean(1)
        ctx = self.ctx_proj(torch.cat([g_node, z_veh_mean, g_graph], dim=-1))
        Q = self.Wq(ctx).unsqueeze(1)
        logits = torch.bmm(Q, self.Wk(Z_node).transpose(1, 2)).squeeze(1) / self._scale
        logits = self.clip * torch.tanh(logits)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class VehicleDecoder(nn.Module):
    def __init__(
        self,
        D: int,
        clip: float = 10.0,
        context_hidden_dim: int = 128,
        context_dropout: float = 0.0,
    ):
        super().__init__()
        self.clip = clip
        layers = [nn.Linear(D * 3, context_hidden_dim), nn.ReLU()]
        if context_dropout > 0:
            layers.append(nn.Dropout(context_dropout))
        self.ctx_proj = nn.Sequential(*layers)
        self.Wq = nn.Linear(context_hidden_dim, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self._scale: Optional[float] = None

    def forward(
        self,
        g_veh: torch.Tensor,
        Z_node: torch.Tensor,
        g_graph: torch.Tensor,
        Z_veh: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._scale is None:
            self._scale = math.sqrt(Z_veh.shape[-1])
        z_node_mean = Z_node.mean(1)
        ctx = self.ctx_proj(torch.cat([g_veh, z_node_mean, g_graph], dim=-1))
        Q = self.Wq(ctx).unsqueeze(1)
        logits = torch.bmm(Q, self.Wk(Z_veh).transpose(1, 2)).squeeze(1) / self._scale
        logits = self.clip * torch.tanh(logits)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


# ---------------------------------------------------------------------------
# GEMAN - Graph-enhanced multi-relational Attention Network
# ---------------------------------------------------------------------------


class GEMANActorCritic(ActorCritic):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Extract encoder and decoder configs
        encoder_cfg = cfg.get("encoder", {})
        decoder_cfg = cfg.get("decoder", {})
        value_head_cfg = cfg.get("value_head", {})
        node_enc_cfg = encoder_cfg.get("node_encoder", {})
        veh_enc_cfg = encoder_cfg.get("vehicle_encoder", {})
        graph_enc_cfg = encoder_cfg.get("graph_encoder", {})
        node_dec_cfg = decoder_cfg.get("node_decoder", {})
        veh_dec_cfg = decoder_cfg.get("vehicle_decoder", {})

        # Read directly from current config structure
        D: int = int(node_enc_cfg.get("embedding_dim", 128))
        H: int = int(node_enc_cfg.get("n_heads", 4))
        drop: float = float(node_enc_cfg.get("dropout", 0.1))
        use_in: bool = bool(node_enc_cfg.get("use_instance_norm", True))
        clip: float = float(node_dec_cfg.get("clip", 10.0))

        # Encoder layers: read per-encoder n_layers
        n_node_layers: int = int(node_enc_cfg.get("n_layers", 3))
        n_veh_layers: int = int(veh_enc_cfg.get("n_layers", 3))

        # Decoder context projection parameters (new config options)
        node_ctx_hidden = int(node_dec_cfg.get("context_hidden_dim", 128))
        node_ctx_dropout = float(node_dec_cfg.get("context_dropout", 0.0))
        veh_ctx_hidden = int(veh_dec_cfg.get("context_hidden_dim", 128))
        veh_ctx_dropout = float(veh_dec_cfg.get("context_dropout", 0.0))

        self.node_encoder = NodeEncoder(D, H, n_node_layers, drop, use_in)
        self.vehicle_encoder = VehicleEncoder(D, H, n_veh_layers, drop, use_in)

        # Build graph encoder with merged config (name + hyperparams from registry)
        graph_encoder_cls = graph_enc_cfg.get("name")
        self.graph_encoder = graph_encoder_cls(graph_enc_cfg)
        self.node_decoder = NodeDecoder(
            D, clip, context_hidden_dim=node_ctx_hidden, context_dropout=node_ctx_dropout
        )
        self.vehicle_decoder = VehicleDecoder(
            D, clip, context_hidden_dim=veh_ctx_hidden, context_dropout=veh_ctx_dropout
        )

        # Build configurable value head from config
        value_hidden_dims = value_head_cfg.get("hidden_dims", [D, D])
        value_use_dropout = value_head_cfg.get("use_dropout", True)
        value_dropout = float(value_head_cfg.get("dropout", 0.1))

        self.value_head = ValueHead(
            input_dim=D * 3,
            hidden_dims=value_hidden_dims,
            dropout=value_dropout,
            use_dropout=value_use_dropout,
        )

        ortho_init: bool = (
            cfg.get("regularization", {}).get("ortho_init", True)
            if isinstance(cfg.get("regularization"), dict)
            else cfg.get("ortho_init", True)
        )
        if ortho_init:
            self._ortho_init(self)

    @classmethod
    def from_config(cls, cfg: Dict) -> "GEMANActorCritic":
        """Factory method: instantiate GEMANActorCritic from config dict.

        Supports both old flat structure and new hierarchical structure with network key.
        """
        # Support both old (flat) and new (with network key) structure
        network_cfg = cfg.get("layers", cfg)
        return cls(cfg=network_cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(x, device: str, dtype=torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def _node_mask(action_mask: torch.Tensor, N1: int, V2K: int) -> torch.Tensor:
        return action_mask.view(action_mask.shape[0], N1, V2K).any(dim=-1)

    @staticmethod
    def _veh_mask(action_mask: torch.Tensor, N1: int, V2K: int) -> torch.Tensor:
        return action_mask.view(action_mask.shape[0], N1, V2K).any(dim=1)

    # ------------------------------------------------------------------
    # Observation unpacking — arrays only, no preprocessing
    # ------------------------------------------------------------------

    def _unpack(self, obs: Dict, device: str):
        """
        Convert obs dict values to tensors on the correct device.
        All arrays are already normalised by state_to_obs.
        Adds batch dimension if missing.
        """
        nf = self._to_tensor(
            obs["node_features"], device, torch.float32
        )  # (B, N+1, 5)  or (N+1, 5)
        vf = self._to_tensor(
            obs["vehicle_features"], device, torch.float32
        )  # (B, 2K,  5)  or (2K,  5)
        t_ei = self._to_tensor(obs["truck_edge_index"], device, torch.long)  # (2, E)
        t_ea = self._to_tensor(obs["truck_edge_attr"], device, torch.float32)  # (E, 2)
        d_ei = self._to_tensor(obs["drone_edge_index"], device, torch.long)  # (2, E)
        d_ea = self._to_tensor(obs["drone_edge_attr"], device, torch.float32)  # (E, 2)

        # Topology is shared across the batch; callers may still store it as
        # (1, 2, E) or (B, 2, E). Collapse that to the expected (2, E) form.
        if t_ei.dim() == 3:
            t_ei = t_ei[0]
        if d_ei.dim() == 3:
            d_ei = d_ei[0]

        if nf.dim() == 2:
            nf = nf.unsqueeze(0)
            vf = vf.unsqueeze(0)
        # edge tensors stay 2-D (shared across batch)

        return nf, vf, t_ei, t_ea, d_ei, d_ea

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, obs: Dict, device: str):
        nf, vf, t_ei, t_ea, d_ei, d_ea = self._unpack(obs, device)
        B, N1, _ = nf.shape
        V2K = vf.shape[1]

        Z_node, g_node = self.node_encoder(nf)
        Z_veh, g_veh = self.vehicle_encoder(vf)

        # GraphEncoder receives [x, y, tw_open, tw_close] only — demand is a
        # delivery semantic, not a routing-structure signal.  Dropping it keeps
        # the graph encoder focused on spatial-temporal topology.
        # nf format: [x, y, linehaul_demand, backhaul_demand, tw_open, tw_close]
        # Extract indices [0, 1, 4, 5] = [x, y, tw_open, tw_close]
        nf_gnn = torch.cat([nf[:, :, :2], nf[:, :, 4:]], dim=-1)  # (B, N+1, 4)
        Z_graph, g_graph = self.graph_encoder(nf_gnn, t_ei, t_ea, d_ei, d_ea)

        return Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K

    # ------------------------------------------------------------------
    # BaseNetwork interface
    # ------------------------------------------------------------------

    def forward(self, obs, action_mask=None, context=None):
        device = globals.DEVICE
        Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K = self._encode(
            obs, device
        )

        # Convert action_mask to torch tensor if it's numpy
        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            action_mask = torch.from_numpy(action_mask).to(device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        n_mask = (
            self._node_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        v_mask = (
            self._veh_mask(action_mask, N1, V2K) if action_mask is not None else None
        )

        ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)  # (B, N1)
        lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)  # (B, V2K)

        flat = (ln.unsqueeze(2) + lv.unsqueeze(1)).view(ln.shape[0], N1 * V2K)
        if action_mask is not None:
            flat = flat.masked_fill(~action_mask, float("-inf"))

        # Ensure all features are 2D (B, D) before concatenation
        g_node_2d = g_node if g_node.dim() == 2 else g_node.flatten(1)
        g_veh_2d = g_veh if g_veh.dim() == 2 else g_veh.flatten(1)
        g_graph_2d = g_graph if g_graph.dim() == 2 else g_graph.flatten(1)
        value = self.value_head(
            torch.cat([g_node_2d, g_veh_2d, g_graph_2d], dim=-1)
        ).squeeze(-1)
        return flat, value

    def evaluate(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        context=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute logits/values or evaluate given actions."""
        device = globals.DEVICE
        Z_node, g_node, Z_veh, g_veh, Z_graph, g_graph, N1, V2K = self._encode(
            obs, device
        )

        # Convert action_mask to torch tensor if it's numpy
        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            action_mask = torch.from_numpy(action_mask).to(device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        n_mask = (
            self._node_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        v_mask = (
            self._veh_mask(action_mask, N1, V2K) if action_mask is not None else None
        )

        ln = self.node_decoder(g_node, Z_veh, g_graph, Z_node, n_mask)
        lv = self.vehicle_decoder(g_veh, Z_node, g_graph, Z_veh, v_mask)

        # Joint logits over all (node, vehicle) pairs, masked to feasible pairs.
        # Sampling and log-prob use the single joint Categorical, so the policy
        # correctly represents p(node, vehicle | feasible) rather than the
        # factorized approximation p_node * p_vehicle which ignores infeasible pairs.
        flat_l = (ln.unsqueeze(2) + lv.unsqueeze(1)).view(ln.shape[0], N1 * V2K)
        if action_mask is not None:
            flat_l = flat_l.masked_fill(~action_mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=flat_l)
        entropy = dist.entropy()

        # Ensure all features are 2D (B, D) before concatenation
        g_node_2d = g_node if g_node.dim() == 2 else g_node.flatten(1)
        g_veh_2d = g_veh if g_veh.dim() == 2 else g_veh.flatten(1)
        g_graph_2d = g_graph if g_graph.dim() == 2 else g_graph.flatten(1)
        value = self.value_head(
            torch.cat([g_node_2d, g_veh_2d, g_graph_2d], dim=-1)
        ).squeeze(-1)

        if actions is None:
            return flat_l, value, entropy
        else:
            log_probs = dist.log_prob(actions)
            return log_probs, value, entropy
