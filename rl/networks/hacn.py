"""
networks/hacn.py
----------------
Heterogeneous Attention Construction Network (HACN) for VRPBTW.

Imports feature-dimension constants from problems/vrpbtw.py so the
network knows its input sizes without depending on the full problem module.

Architecture overview
---------------------
ENCODER  (once per instance)
  Node features (N+1, NODE_FEAT_DIM)
  → linear projection → L × [Self-attn + Heterogeneous cross-attn + FF + Norm]
  → Z_node (N+1, D)

VEHICLE GNN  (every step, variable graph)
  Visited-node features + edge-conditioned message passing
  → Z_veh (2K, D)

DECODER  (every step, hierarchical two-pointer)
  Upper: node selection    log π_U(n*)
  Lower: vehicle selection log π_L(v* | n*)
  Joint log-prob = log π_U + log π_L

Action encoding:  flat = node * 2K + vehicle_idx
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from core.module import BaseNetwork, _MHA, _FF, _make_norm
from problems.vrpbtw import NODE_FEAT_DIM, VEH_FEAT_DIM, EDGE_FEAT_DIM


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------


class _EncoderBlock(nn.Module):
    def __init__(self, D: int, H: int, dropout: float, use_in: bool):
        super().__init__()
        self.sa = _MHA(D, H, dropout)
        self.het_l2b = _MHA(D, H, dropout)
        self.het_b2l = _MHA(D, H, dropout)
        self.ff = _FF(D, dropout)
        self.norm1 = _make_norm(use_in, D)
        self.norm2 = _make_norm(use_in, D)

    def forward(
        self,
        h: torch.Tensor,  # (B, N+1, D)
        h_l: torch.Tensor,  # (B, M, D)
        h_b: torch.Tensor,  # (B, P, D)
        l_idx: torch.Tensor,  # (M,)
        b_idx: torch.Tensor,  # (P,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_sa = self.sa(h, h, h)
        h_het = torch.zeros_like(h)
        if h_l.shape[1] > 0 and h_b.shape[1] > 0:
            h_het[:, l_idx] = self.het_l2b(h_l, h_b, h_b)
            h_het[:, b_idx] = self.het_b2l(h_b, h_l, h_l)
        h = self.norm1(h + h_sa + h_het)
        h = self.norm2(h + self.ff(h))
        h_l = h[:, l_idx] if h_l.shape[1] > 0 else h_l
        h_b = h[:, b_idx] if h_b.shape[1] > 0 else h_b
        return h, h_l, h_b


# ---------------------------------------------------------------------------
# Edge-conditioned message passing layer
# ---------------------------------------------------------------------------


class _EdgeConvMessage(nn.Module):
    def __init__(self, D: int, edge_feat_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(D + edge_feat_dim, D),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, D),
        )
        self.norm = nn.LayerNorm(D)

    def forward(
        self,
        h: torch.Tensor,  # (V, D)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, edge_feat_dim)
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        msg = self.msg_mlp(torch.cat([h[src], edge_attr], dim=-1))
        V = h.shape[0]
        agg = torch.zeros(V, h.shape[1], device=h.device, dtype=h.dtype)
        count = torch.zeros(V, 1, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        count.scatter_add_(
            0,
            dst.unsqueeze(1),
            torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype),
        )
        return self.norm(h + agg / count.clamp(min=1.0))


# ---------------------------------------------------------------------------
# Vehicle GNN
# ---------------------------------------------------------------------------


class _VehicleGNN(nn.Module):
    def __init__(
        self,
        D: int,
        K: int,
        edge_feat_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.D = D
        self.K = K
        node_in = D + 1 + K + 1  # Z_node + visit_time + fleet_onehot + vtype

        self.node_proj = nn.Linear(node_in, D)
        self.layers = nn.ModuleList(
            [_EdgeConvMessage(D, edge_feat_dim, dropout) for _ in range(n_layers)]
        )
        self.props_mlp = nn.Sequential(
            nn.Linear(VEH_FEAT_DIM, D), nn.ReLU(), nn.Linear(D, D)
        )
        self.readout_norm = nn.LayerNorm(D)

    def forward(
        self,
        Z_node_full: torch.Tensor,  # (N+1, D)
        veh_feat: torch.Tensor,  # (2K, VEH_FEAT_DIM)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, EDGE_FEAT_DIM)
        visited_nodes: List[int],
        visit_meta: torch.Tensor,  # (|V_t|, 1+K+1)
        current_nodes: List[int],  # (2K,)
    ) -> torch.Tensor:
        D, device = self.D, Z_node_full.device
        props = self.props_mlp(veh_feat)  # (2K, D)

        if len(visited_nodes) == 0:
            zero_base = torch.zeros(len(current_nodes), D, device=device)
            return self.readout_norm(zero_base + props)

        v_idx = torch.tensor(visited_nodes, dtype=torch.long, device=device)
        h_in = torch.cat([Z_node_full[v_idx], visit_meta], dim=-1)
        h = self.node_proj(h_in)

        local_map = {g: l for l, g in enumerate(visited_nodes)}

        if edge_index.shape[1] > 0:
            src_g, dst_g = edge_index[0].tolist(), edge_index[1].tolist()
            valid = [
                (s, d)
                for s, d in zip(src_g, dst_g)
                if s in local_map and d in local_map
            ]
            if valid:
                src_l = torch.tensor(
                    [local_map[s] for s, _ in valid], dtype=torch.long, device=device
                )
                dst_l = torch.tensor(
                    [local_map[d] for _, d in valid], dtype=torch.long, device=device
                )
                e_idx_l = torch.stack([src_l, dst_l], dim=0)
                keep = torch.tensor(
                    [
                        i
                        for i, (s, d) in enumerate(zip(src_g, dst_g))
                        if s in local_map and d in local_map
                    ],
                    dtype=torch.long,
                    device=device,
                )
                e_attr_l = edge_attr[keep]
            else:
                e_idx_l = torch.zeros(2, 0, dtype=torch.long, device=device)
                e_attr_l = torch.zeros(0, edge_attr.shape[-1], device=device)
        else:
            e_idx_l = torch.zeros(2, 0, dtype=torch.long, device=device)
            e_attr_l = torch.zeros(0, edge_attr.shape[-1], device=device)

        for layer in self.layers:
            h = layer(h, e_idx_l, e_attr_l)

        Z_veh = torch.zeros(len(current_nodes), D, device=device)
        for vi, cn in enumerate(current_nodes):
            if cn in local_map:
                Z_veh[vi] = h[local_map[cn]]
        return self.readout_norm(Z_veh + props)


# ---------------------------------------------------------------------------
# PolicyNetwork  (HACN)
# ---------------------------------------------------------------------------


class PolicyNetwork(BaseNetwork):
    """
    Heterogeneous Attention Construction Network for VRPBTW.

    Parameters
    ----------
    obs_shape : (N+1, NODE_FEAT_DIM)
    cfg       : NetworkConfig
    n_fleets  : K (number of truck-drone fleet pairs)
    """

    def __init__(self, obs_shape: Tuple[int, ...], cfg, n_fleets: int = 2):
        super().__init__()
        self.cfg = cfg
        self.obs_shape = obs_shape
        self.K = n_fleets

        D, H, L = cfg.embed_dim, cfg.n_heads, cfg.n_encoder_layers
        drop = cfg.dropout
        use_in = getattr(cfg, "use_instance_norm", True)

        # Encoder
        self.node_embed = nn.Linear(NODE_FEAT_DIM, D)
        self.enc_blocks = nn.ModuleList(
            [_EncoderBlock(D, H, drop, use_in) for _ in range(L)]
        )

        # Vehicle GNN
        self.vehicle_gnn = _VehicleGNN(
            D=D, K=n_fleets, edge_feat_dim=EDGE_FEAT_DIM, n_layers=2, dropout=drop
        )

        # Decoder — upper (node)
        self.ctx_upper = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU())
        self.Wq_upper = nn.Linear(D, D, bias=False)
        self.Wk_upper = nn.Linear(D, D, bias=False)

        # Decoder — lower (vehicle)
        self.ctx_lower = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU())
        self.score_lower = nn.Sequential(
            nn.Linear(D * 2, D), nn.ReLU(), nn.Linear(D, 1)
        )

        # Value head
        self.value_head = nn.Sequential(nn.Linear(D * 2, D), nn.Tanh(), nn.Linear(D, 1))

        if cfg.ortho_init:
            self._ortho_init(self)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        node_feat: torch.Tensor,  # (B, N+1, NODE_FEAT_DIM)
        l_idx: torch.Tensor,
        b_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node_embed(node_feat)
        h_l = h[:, l_idx] if l_idx.numel() > 0 else h[:, :0]
        h_b = h[:, b_idx] if b_idx.numel() > 0 else h[:, :0]
        for blk in self.enc_blocks:
            h, h_l, h_b = blk(h, h_l, h_b, l_idx, b_idx)
        return h, h.mean(dim=1)

    # ------------------------------------------------------------------
    # Prep batch from obs dict
    # ------------------------------------------------------------------

    def _prep_batch(self, obs: Dict, device: str):
        nf = torch.FloatTensor(obs["node_features"]).to(device)
        vf = torch.FloatTensor(obs["vehicle_features"]).to(device)
        ei, ea, ef = obs["edge_index"], obs["edge_attr"], obs["edge_fleet"]
        if nf.dim() == 2:
            nf = nf.unsqueeze(0)
            vf = vf.unsqueeze(0)
            ei, ea, ef = [ei], [ea], [ef]
        ei_t = [torch.LongTensor(np.array(e, dtype=np.int64)).to(device) for e in ei]
        ea_t = [torch.FloatTensor(np.array(a, dtype=np.float32)).to(device) for a in ea]
        ef_t = [torch.LongTensor(np.array(f, dtype=np.int64)).to(device) for f in ef]
        return nf, vf, ei_t, ea_t, ef_t

    def _node_indices(self, node_feat: torch.Tensor):
        demand = node_feat[0, :, 2]
        return torch.where(demand > 0)[0], torch.where(demand < 0)[0]

    @staticmethod
    def _visited_and_meta(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_fleet: torch.Tensor,
        K: int,
        device: str,
    ) -> Tuple[List[int], torch.Tensor]:
        if edge_index.shape[1] == 0:
            return [], torch.zeros(0, 1 + K + 1, device=device)
        all_nodes = sorted(torch.cat([edge_index[0], edge_index[1]]).unique().tolist())
        V = len(all_nodes)
        meta = torch.zeros(V, 1 + K + 1, device=device)
        local_map = {int(g): l for l, g in enumerate(all_nodes)}
        for ei_idx, dst_g in enumerate(edge_index[1].tolist()):
            dst_g = int(dst_g)
            if dst_g not in local_map:
                continue
            li = local_map[dst_g]
            arr_t = float(edge_attr[ei_idx, 4].item())
            fleet_id = int(edge_fleet[ei_idx].item())
            vtype = float(edge_attr[ei_idx, 0].item())
            meta[li, 0] = arr_t
            meta[li, 1 : 1 + K] = 0.0
            if 0 <= fleet_id < K:
                meta[li, 1 + fleet_id] = 1.0
            meta[li, 1 + K] = vtype
        return all_nodes, meta

    @staticmethod
    def _upper_mask(action_mask: torch.Tensor, N1: int, V2K: int) -> torch.Tensor:
        return action_mask.view(action_mask.shape[0], N1, V2K).any(dim=-1)

    @staticmethod
    def _lower_mask(
        action_mask: torch.Tensor, node_id: torch.Tensor, N1: int, V2K: int
    ) -> torch.Tensor:
        B = action_mask.shape[0]
        m = action_mask.view(B, N1, V2K)
        idx = node_id.view(B, 1, 1).expand(B, 1, V2K)
        return m.gather(1, idx).squeeze(1)

    def _decode_upper(
        self, graph_node, graph_veh, Z_node, node_mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = self.ctx_upper(torch.cat([graph_node, graph_veh], dim=-1))
        query = self.Wq_upper(ctx).unsqueeze(1)
        keys = self.Wk_upper(Z_node)
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(
            self.cfg.embed_dim
        )
        logits = self.cfg.clip_logits * torch.tanh(logits)
        logits = self._apply_mask(logits, node_mask)
        return logits, ctx

    def _decode_lower(self, graph_veh, Z_node_sel, Z_veh, veh_mask) -> torch.Tensor:
        ctx = self.ctx_lower(torch.cat([graph_veh, Z_node_sel], dim=-1))
        ctx_e = ctx.unsqueeze(1).expand_as(Z_veh)
        logits = self.score_lower(torch.cat([Z_veh, ctx_e], dim=-1)).squeeze(-1)
        return self._apply_mask(logits, veh_mask)

    def _encode_and_gnn(self, obs: Dict, device: str):
        nf, vf, ei_t, ea_t, ef_t = self._prep_batch(obs, device)
        B, N1, _ = nf.shape
        V2K = vf.shape[1]
        l_idx, b_idx = self._node_indices(nf)
        Z_node, graph_node = self.encode(nf, l_idx, b_idx)

        Z_veh_list = []
        for b in range(B):
            visited, v_meta = self._visited_and_meta(
                ei_t[b], ea_t[b], ef_t[b], self.K, device
            )
            cur_nodes = [
                int(round(float(vf[b, vi, 0].item()) * (N1 - 1))) for vi in range(V2K)
            ]
            Z_veh_list.append(
                self.vehicle_gnn(
                    Z_node[b], vf[b], ei_t[b], ea_t[b], visited, v_meta, cur_nodes
                )
            )
        Z_veh = torch.stack(Z_veh_list, dim=0)
        graph_veh = Z_veh.mean(dim=1)
        return Z_node, graph_node, Z_veh, graph_veh, N1, V2K

    # ------------------------------------------------------------------
    # BaseNetwork interface
    # ------------------------------------------------------------------

    def forward(self, obs: Dict, action_mask=None, context=None):
        device = next(self.parameters()).device.type
        Z_node, graph_node, Z_veh, graph_veh, N1, V2K = self._encode_and_gnn(
            obs, device
        )
        B = Z_node.shape[0]

        node_mask = (
            self._upper_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        logits_U, ctx_U = self._decode_upper(graph_node, graph_veh, Z_node, node_mask)
        flat_logits = self._build_flat_logits(
            logits_U, Z_node, graph_veh, Z_veh, action_mask, N1, V2K, device
        )
        value = self.value_head(
            torch.cat([ctx_U.detach(), graph_veh.detach()], dim=-1)
        ).squeeze(-1)
        return flat_logits, value

    def get_action_and_log_prob(
        self, obs: Dict, action_mask=None, context=None, deterministic=False
    ):
        device = next(self.parameters()).device.type
        Z_node, graph_node, Z_veh, graph_veh, N1, V2K = self._encode_and_gnn(
            obs, device
        )
        B = Z_node.shape[0]

        node_mask = (
            self._upper_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        logits_U, ctx_U = self._decode_upper(graph_node, graph_veh, Z_node, node_mask)
        dist_U = torch.distributions.Categorical(logits=logits_U)
        node_id = logits_U.argmax(-1) if deterministic else dist_U.sample()
        lp_node = dist_U.log_prob(node_id)

        Z_node_sel = Z_node[torch.arange(B, device=device), node_id]
        veh_mask = (
            self._lower_mask(action_mask, node_id, N1, V2K)
            if action_mask is not None
            else None
        )
        logits_L = self._decode_lower(graph_veh, Z_node_sel, Z_veh, veh_mask)
        dist_L = torch.distributions.Categorical(logits=logits_L)
        veh_id = logits_L.argmax(-1) if deterministic else dist_L.sample()
        lp_veh = dist_L.log_prob(veh_id)

        value = self.value_head(
            torch.cat([ctx_U.detach(), graph_veh.detach()], dim=-1)
        ).squeeze(-1)
        return node_id * V2K + veh_id, lp_node + lp_veh, value

    def evaluate_actions(
        self, obs: Dict, actions: torch.Tensor, action_mask=None, context=None
    ):
        device = next(self.parameters()).device.type
        Z_node, graph_node, Z_veh, graph_veh, N1, V2K = self._encode_and_gnn(
            obs, device
        )
        B = Z_node.shape[0]

        node_id = actions // V2K
        veh_id = actions % V2K

        node_mask = (
            self._upper_mask(action_mask, N1, V2K) if action_mask is not None else None
        )
        logits_U, ctx_U = self._decode_upper(graph_node, graph_veh, Z_node, node_mask)
        dist_U = torch.distributions.Categorical(logits=logits_U)
        lp_node = dist_U.log_prob(node_id)
        ent_U = dist_U.entropy()

        Z_node_sel = Z_node[torch.arange(B, device=device), node_id]
        veh_mask = (
            self._lower_mask(action_mask, node_id, N1, V2K)
            if action_mask is not None
            else None
        )
        logits_L = self._decode_lower(graph_veh, Z_node_sel, Z_veh, veh_mask)
        dist_L = torch.distributions.Categorical(logits=logits_L)
        lp_veh = dist_L.log_prob(veh_id)
        ent_L = dist_L.entropy()

        value = self.value_head(
            torch.cat([ctx_U.detach(), graph_veh.detach()], dim=-1)
        ).squeeze(-1)
        return lp_node + lp_veh, value, ent_U + ent_L

    def _build_flat_logits(
        self, logits_U, Z_node, graph_veh, Z_veh, action_mask, N1, V2K, device
    ) -> torch.Tensor:
        B = logits_U.shape[0]
        flat = torch.full((B, N1 * V2K), float("-inf"), device=logits_U.device)
        for n in range(N1):
            Z_n = Z_node[:, n, :]
            v_mask = None
            if action_mask is not None:
                nid_t = torch.full((B,), n, dtype=torch.long, device=logits_U.device)
                v_mask = self._lower_mask(action_mask, nid_t, N1, V2K)
            l_L = self._decode_lower(graph_veh, Z_n, Z_veh, v_mask)
            for v in range(V2K):
                flat[:, n * V2K + v] = logits_U[:, n] + l_L[:, v]
        if action_mask is not None:
            flat = flat.masked_fill(~action_mask, float("-inf"))
        return flat
