"""
impl/am.py
----------
Attention Model (AM) for VRPBTW.

Architecture:
  Encoder:
    - NodeEncoder: Multi-head attention over node features → Z_node (B, N+1, D), g_node (B, D)
    - VehicleEncoder: Multi-head attention over vehicle features → Z_veh (B, 2K, D), g_veh (B, D)

  Decoder:
    - Single attention decoder: selects next node from unserved nodes
    - Context: pooled embeddings from node and vehicle encoders
    - Output: logits over N+1 nodes (action_mask handles feasibility)

  Value head: MLP(g_node ‖ g_veh)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import globals
from core.network import ActorCritic, _MHA, _FF, _make_norm
from impl.vrpbtw import NODE_FEAT_DIM, VEH_FEAT_DIM


# ---------------------------------------------------------------------------
# Encoders (reuse from GEMAN)
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
        w = torch.softmax(self.pool(h), dim=1)
        return h, (w * h).sum(dim=1)


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
        w = torch.softmax(self.pool(h), dim=1)
        return h, (w * h).sum(dim=1)


# ---------------------------------------------------------------------------
# Single Node Decoder (AM-style)
# ---------------------------------------------------------------------------


class NodeDecoder(nn.Module):
    """Attention-based decoder for selecting next node.

    Args:
        D: Embedding dimension
        clip: Tanh clipping range for logits
        context_hidden_dim: Hidden layer dimension for context projection
        context_dropout: Dropout rate for context projection
    """

    def __init__(
        self,
        D: int,
        clip: float = 10.0,
        context_hidden_dim: int = 128,
        context_dropout: float = 0.0,
    ):
        super().__init__()
        self.clip = clip
        layers = [nn.Linear(D * 2, context_hidden_dim), nn.ReLU()]
        if context_dropout > 0:
            layers.append(nn.Dropout(context_dropout))
        self.ctx_proj = nn.Sequential(*layers)
        self.Wq = nn.Linear(context_hidden_dim, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self._scale: Optional[float] = None

    def forward(
        self,
        g_node: torch.Tensor,
        g_veh: torch.Tensor,
        Z_node: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for next node selection.

        Args:
            g_node: (B, D) global node embedding
            g_veh: (B, D) global vehicle embedding
            Z_node: (B, N+1, D) node embeddings
            mask: (B, N+1) boolean mask (True = selectable)

        Returns:
            logits: (B, N+1) unnormalized scores
        """
        if self._scale is None:
            self._scale = math.sqrt(Z_node.shape[-1])

        # Combine context from node and vehicle encoders
        ctx = self.ctx_proj(torch.cat([g_node, g_veh], dim=-1))
        Q = self.Wq(ctx).unsqueeze(1)  # (B, 1, D)
        logits = torch.bmm(Q, self.Wk(Z_node).transpose(1, 2)).squeeze(1) / self._scale
        logits = self.clip * torch.tanh(logits)

        if mask is not None:
            # Handle mask shape: ensure it's (B, N+1)
            B = logits.shape[0]
            if mask.dim() == 1:
                # mask is (N+1,), expand to (B, N+1)
                mask = mask.unsqueeze(0).expand(B, -1)
            elif mask.shape[0] != B and mask.shape[0] == 1:
                # mask is (1, N+1) but B > 1, broadcast it
                mask = mask.expand(B, -1)

            logits = logits.masked_fill(~mask, float("-inf"))

        return logits


# ---------------------------------------------------------------------------
# Value Head
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
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
        return self.net(x)


# ---------------------------------------------------------------------------
# AM Actor-Critic
# ---------------------------------------------------------------------------


class AMActorCritic(ActorCritic):
    """Attention Model Actor-Critic for sequential vehicle routing."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        encoder_cfg = cfg.get("encoder", {})
        decoder_cfg = cfg.get("decoder", {})
        value_head_cfg = cfg.get("value_head", {})
        node_enc_cfg = encoder_cfg.get("node_encoder", {})
        veh_enc_cfg = encoder_cfg.get("vehicle_encoder", {})
        graph_enc_cfg = encoder_cfg.get("graph_encoder", {})
        node_dec_cfg = decoder_cfg.get("node_decoder", {})

        # Hyperparameters
        D: int = int(node_enc_cfg.get("embedding_dim", 128))
        H: int = int(node_enc_cfg.get("n_heads", 4))
        drop: float = float(node_enc_cfg.get("dropout", 0.1))
        use_in: bool = bool(node_enc_cfg.get("use_instance_norm", True))
        clip: float = float(node_dec_cfg.get("clip", 10.0))

        n_node_layers: int = int(node_enc_cfg.get("n_layers", 3))
        n_veh_layers: int = int(veh_enc_cfg.get("n_layers", 3))

        node_ctx_hidden = int(node_dec_cfg.get("context_hidden_dim", 128))
        node_ctx_dropout = float(node_dec_cfg.get("context_dropout", 0.0))

        self.node_encoder = NodeEncoder(D, H, n_node_layers, drop, use_in)
        self.vehicle_encoder = VehicleEncoder(D, H, n_veh_layers, drop, use_in)

        self.node_decoder = NodeDecoder(
            D, clip, context_hidden_dim=node_ctx_hidden, context_dropout=node_ctx_dropout
        )

        # Value head
        value_hidden_dims = value_head_cfg.get("hidden_dims", [D, D])
        value_use_dropout = value_head_cfg.get("use_dropout", True)
        value_dropout = float(value_head_cfg.get("dropout", 0.1))

        self.value_head = ValueHead(
            input_dim=D * 2,
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
    def from_config(cls, cfg: Dict) -> "AMActorCritic":
        """Factory method: instantiate AMActorCritic from config dict."""
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

    # ------------------------------------------------------------------
    # Observation unpacking
    # ------------------------------------------------------------------

    def _unpack(self, obs: Dict, device: str, vehicle_idx: int = None):
        """Convert obs dict values to tensors on the correct device.

        Args:
            obs: Observation dict
            device: Target device
            vehicle_idx: If provided, extract only this vehicle's features (for sequential routing)
        """
        nf = self._to_tensor(obs["node_features"], device, torch.float32)
        vf = self._to_tensor(obs["vehicle_features"], device, torch.float32)
        t_ei = self._to_tensor(obs["truck_edge_index"], device, torch.long)
        t_ea = self._to_tensor(obs["truck_edge_attr"], device, torch.float32)
        d_ei = self._to_tensor(obs["drone_edge_index"], device, torch.long)
        d_ea = self._to_tensor(obs["drone_edge_attr"], device, torch.float32)

        if t_ei.dim() == 3:
            t_ei = t_ei[0]
        if d_ei.dim() == 3:
            d_ei = d_ei[0]

        if nf.dim() == 2:
            nf = nf.unsqueeze(0)
            vf = vf.unsqueeze(0)

        # For sequential routing: extract only current vehicle
        if vehicle_idx is not None:
            if vf.dim() == 3:  # (B, 2K, 6)
                vf = vf[:, vehicle_idx:vehicle_idx+1, :]  # (B, 1, 6)
            elif vf.dim() == 2:  # (2K, 6)
                vf = vf[vehicle_idx:vehicle_idx+1, :]  # (1, 6)

        return nf, vf, t_ei, t_ea, d_ei, d_ea

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, obs: Dict, device: str, vehicle_idx: int = None):
        """Encode observation.

        Args:
            obs: Observation dict
            device: Target device
            vehicle_idx: If provided, extract only this vehicle's features (for sequential routing)
        """
        nf, vf, t_ei, t_ea, d_ei, d_ea = self._unpack(obs, device, vehicle_idx)

        Z_node, g_node = self.node_encoder(nf)
        Z_veh, g_veh = self.vehicle_encoder(vf)

        return Z_node, g_node, Z_veh, g_veh

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs, action_mask=None, context=None, vehicle_idx: int = None):
        """Forward pass for sequential vehicle routing.

        Args:
            obs: Observation dict
            action_mask: (B, N+1) boolean mask for feasible nodes
            context: Unused (for API compatibility)
            vehicle_idx: Current vehicle index (0 to 2K-1). If provided, encodes only this vehicle.

        Returns:
            logits: (B, N+1) unnormalized node selection scores
            value: (B,) state value estimate
        """
        device = globals.DEVICE
        Z_node, g_node, Z_veh, g_veh = self._encode(obs, device, vehicle_idx)

        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            action_mask = torch.from_numpy(action_mask).to(device)

        # Handle action_mask shape for sequential routing
        if action_mask is not None:
            B = Z_node.shape[0]
            N1 = Z_node.shape[1]  # Number of nodes (including depot)

            # If mask is bilevel (B, N+1, 2K) from environment, convert to unilevel
            if action_mask.dim() == 3:  # (B, N+1, 2K)
                if vehicle_idx is not None:
                    action_mask = action_mask[:, :, vehicle_idx]  # (B, N+1)
                else:
                    # Extract nodes feasible for any vehicle
                    action_mask = action_mask.any(dim=2)  # (B, N+1) - True if feasible for ≥1 vehicle

            # If mask is flattened bilevel (B, N+1*2K), convert to unilevel
            elif action_mask.dim() == 2 and action_mask.shape[1] > N1:
                # mask is (B, N+1*2K), reshape and extract
                V2K = action_mask.shape[1] // N1
                action_mask = action_mask.view(B, N1, V2K)  # (B, N+1, 2K)
                if vehicle_idx is not None:
                    action_mask = action_mask[:, :, vehicle_idx]  # (B, N+1)
                else:
                    action_mask = action_mask.any(dim=2)  # (B, N+1)

            # Standard case: already (B, N+1) or (N+1,)
            elif action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)  # (N+1,) → (1, N+1)

        n_mask = action_mask
        logits = self.node_decoder(g_node, g_veh, Z_node, n_mask)

        # Value estimation
        g_node_2d = g_node if g_node.dim() == 2 else g_node.flatten(1)
        g_veh_2d = g_veh if g_veh.dim() == 2 else g_veh.flatten(1)
        value = self.value_head(
            torch.cat([g_node_2d, g_veh_2d], dim=-1)
        ).squeeze(-1)

        return logits, value

    def evaluate(
        self,
        obs,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        context=None,
        vehicle_idx: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute logits/values or evaluate given actions.

        Args:
            obs: Observation dict
            action_mask: (B, N+1) boolean mask
            actions: (B,) node indices to evaluate (if None, returns logits)
            context: Unused
            vehicle_idx: Current vehicle index (0 to 2K-1)

        Returns:
            log_probs or logits: (B,) or (B, N+1)
            value: (B,)
            entropy: (B,)
        """
        device = globals.DEVICE
        Z_node, g_node, Z_veh, g_veh = self._encode(obs, device, vehicle_idx)

        if action_mask is not None and not isinstance(action_mask, torch.Tensor):
            action_mask = torch.from_numpy(action_mask).to(device)

        # Handle action_mask shape for sequential routing
        if action_mask is not None:
            B = Z_node.shape[0]
            N1 = Z_node.shape[1]  # Number of nodes (including depot)

            # If mask is bilevel (B, N+1, 2K) from environment, convert to unilevel
            if action_mask.dim() == 3:  # (B, N+1, 2K)
                if vehicle_idx is not None:
                    action_mask = action_mask[:, :, vehicle_idx]  # (B, N+1)
                else:
                    # Extract nodes feasible for any vehicle
                    action_mask = action_mask.any(dim=2)  # (B, N+1) - True if feasible for ≥1 vehicle

            # If mask is flattened bilevel (B, N+1*2K), convert to unilevel
            elif action_mask.dim() == 2 and action_mask.shape[1] > N1:
                # mask is (B, N+1*2K), reshape and extract
                V2K = action_mask.shape[1] // N1
                action_mask = action_mask.view(B, N1, V2K)  # (B, N+1, 2K)
                if vehicle_idx is not None:
                    action_mask = action_mask[:, :, vehicle_idx]  # (B, N+1)
                else:
                    action_mask = action_mask.any(dim=2)  # (B, N+1)

            # Standard case: already (B, N+1) or (N+1,)
            elif action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)  # (N+1,) → (1, N+1)

        n_mask = action_mask
        logits = self.node_decoder(g_node, g_veh, Z_node, n_mask)

        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy()

        g_node_2d = g_node if g_node.dim() == 2 else g_node.flatten(1)
        g_veh_2d = g_veh if g_veh.dim() == 2 else g_veh.flatten(1)
        value = self.value_head(
            torch.cat([g_node_2d, g_veh_2d], dim=-1)
        ).squeeze(-1)

        if actions is None:
            return logits, value, entropy
        else:
            log_probs = dist.log_prob(actions)
            return log_probs, value, entropy
