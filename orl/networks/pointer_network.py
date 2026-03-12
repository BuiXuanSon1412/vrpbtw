"""
pointer_network.py
------------------
Pointer Network with Transformer encoder for variable-size combinatorial RL.

Key idea: the decoder produces logits via dot-product attention between
a context vector and node embeddings. Output size = N (dynamic), not fixed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from base_network import BaseNetwork


@dataclass
class PointerNetConfig:
    feat_dim: int  # input node feature dimension
    embed_dim: int = 128  # D
    n_heads: int = 8
    n_encoder_layers: int = 3
    dropout: float = 0.0
    clip_logits: float = 10.0  # tanh clipping (from Bello et al.)
    ortho_init: bool = True


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape
        H, Dh = self.n_heads, self.head_dim

        Q = self.q(query).view(B, T_q, H, Dh).transpose(1, 2)
        K = self.k(key).view(B, T_k, H, Dh).transpose(1, 2)
        V = self.v(value).view(B, T_k, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x, x, x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# Encoder — processes variable N nodes
# ---------------------------------------------------------------------------


class TransformerEncoder(nn.Module):
    """
    Takes node features of any size N and produces node embeddings.

    Input  : (B, N, feat_dim)   — N is dynamic
    Output : (B, N, D)          — N preserved, D fixed
    """

    def __init__(self, cfg: PointerNetConfig):
        super().__init__()
        self.node_embed = nn.Linear(cfg.feat_dim, cfg.embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(cfg.embed_dim, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.n_encoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, feat_dim) → (B, N, D)"""
        h = self.node_embed(x)  # linear projection, no size constraint
        for layer in self.layers:
            h = layer(h)
        return h  # (B, N, D)  — N unchanged


# ---------------------------------------------------------------------------
# Pointer Decoder — outputs N logits via dot-product, not Linear(D→N)
# ---------------------------------------------------------------------------


class PointerDecoder(nn.Module):
    """
    Single-head pointer attention.

    At each step:
      context  = f(graph_embedding, step_features)    → (B, 1, D)
      keys     = W_k * node_embeddings               → (B, N, D)
      logits   = tanh(dot(context, keys) / sqrt(D)) * clip   → (B, N)

    N is never hardcoded — logits dimension == number of input nodes.

    References
    ----------
    Vinyals et al. "Pointer Networks" (2015)
    Kool et al.    "Attention, Learn to Solve Routing Problems!" (2019)
    """

    def __init__(self, cfg: PointerNetConfig):
        super().__init__()
        D = cfg.embed_dim
        self.clip = cfg.clip_logits
        self.scale = math.sqrt(D)

        # Project graph context → query
        self.W_q = nn.Linear(D, D, bias=False)

        # Project node embeddings → keys
        self.W_k = nn.Linear(D, D, bias=False)

        # Value head for critic (separate from pointer)
        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,  # (B, N, D)  — encoder output
        step_context: torch.Tensor,  # (B, D)     — current decision state
        action_mask: Optional[torch.Tensor] = None,  # (B, N) bool, True=feasible
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : (B, N)   — one score per node, masked
        value  : (B,)     — critic estimate
        """
        # Query from step context
        query = self.W_q(step_context).unsqueeze(1)  # (B, 1, D)

        # Keys from node embeddings
        keys = self.W_k(node_embeddings)  # (B, N, D)

        # Pointer scores: dot product query × keys
        # Shape: (B, 1, D) × (B, D, N) = (B, 1, N) → squeeze → (B, N)
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / self.scale

        # Tanh clipping for stable gradients (Bello et al. 2016)
        logits = torch.tanh(logits) * self.clip

        # Mask infeasible nodes
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        # Value estimate from graph context
        graph_context = node_embeddings.mean(dim=1)  # (B, D)
        value = self.value_head(graph_context).squeeze(-1)  # (B,)

        return logits, value


# ---------------------------------------------------------------------------
# Full Pointer Network (Encoder + Decoder combined)
# ---------------------------------------------------------------------------


class PointerNetwork(nn.Module):
    """
    Complete encoder-decoder pointer network.

    CRITICALLY different from your current PolicyNetwork:
      - No Linear(D → fixed_N) decoder
      - logits shape = (B, N_active) always matches input
      - Works for N=5, N=10, N=50 with the SAME weights

    Step context
    ------------
    At each decision step, the decoder needs to know "where are we now?"
    The step_context encodes the current partial solution state:
      - For TSP/VRP: embedding of last visited node
      - For VRPBTW:  mean of [truck_embed, drone_embed, remaining_capacity, time]
    This is projected to D and passed to the pointer decoder.

    Usage
    -----
    net = PointerNetwork(cfg)

    # Encode once per instance (N can be anything)
    node_emb = net.encode(obs)               # (B, N, D)

    # Decode step by step
    logits, value = net.decode(node_emb, step_ctx, mask)  # (B, N) logits
    """

    def __init__(self, cfg: PointerNetConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(cfg)
        self.decoder = PointerDecoder(cfg)

        # Step context projector — maps raw step features → D
        # For VRPBTW: step features = [truck_time, truck_load, drone_time,
        #             drone_load, drone_active, served_ratio]  → 6-dim
        # Change step_context_dim to match your problem
        self.step_context_dim = cfg.feat_dim  # override after construction
        self.ctx_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)

        if cfg.ortho_init:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode node features.

        obs : (B, N, feat_dim)   N is variable — no constraint
        returns (B, N, D)
        """
        return self.encoder(obs)

    def decode(
        self,
        node_embeddings: torch.Tensor,  # (B, N, D)
        action_mask: Optional[torch.Tensor] = None,  # (B, N) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode one action.

        Uses mean pooling of node embeddings as the step context.
        For a richer context, call decode_with_context() instead.

        Returns: logits (B, N), value (B,)
        """
        # Default step context = mean of all node embeddings
        step_context = self.ctx_proj(node_embeddings.mean(dim=1))  # (B, D)
        return self.decoder(node_embeddings, step_context, action_mask)

    def decode_with_context(
        self,
        node_embeddings: torch.Tensor,  # (B, N, D)
        step_context: torch.Tensor,  # (B, D)   — custom context
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode with an explicit step context vector.

        For VRPBTW, build step_context from current vehicle states:
            ctx = concat([truck_node_emb, drone_node_emb, load/Q, time/T])
            ctx = Linear(ctx_dim → D)(ctx)
        """
        return self.decoder(node_embeddings, step_context, action_mask)

    def forward(
        self,
        obs: torch.Tensor,  # (B, N, feat_dim)
        action_mask: Optional[torch.Tensor] = None,  # (B, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode."""
        node_emb = self.encode(obs)
        return self.decode(node_emb, action_mask)

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
