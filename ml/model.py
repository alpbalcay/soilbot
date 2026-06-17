"""Bayesian heterogeneous-GraphSAGE for soil-type prediction.

Encoder: per-column embeddings + numerics -> stack of relation-specific SAGE layers
(one weight set per edge type {knn, delaunay, same_geology}, mean-aggregated with native
scatter ops — no torch-scatter/pyg-lib needed). Heads: hierarchical family->code soil-type
classifier + auxiliary drainage classifier.

Every weight layer is a mean-field Gaussian `BayesianLinear` (Bayes-by-Backprop). With
`sample=False` and KL weight 0 it is exactly the deterministic baseline (A1) and its learned
means warm-start the Bayesian run (A2). Geology enters as an INFORMATIVE PRIOR via an additive
per-node logit offset (`prior_logits`) the residual head is pulled toward by KL (A3); zeroing it
is the prior-ablation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """Mean-field Gaussian linear layer (reparameterization trick)."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # start with small variance so the deterministic warm-start is faithful
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.constant_(self.bias_rho, -5.0)

    def _sigma(self, rho):
        return F.softplus(rho)

    def forward(self, x, sample: bool = True):
        if sample and self.training:
            w = self.weight_mu + self._sigma(self.weight_rho) * torch.randn_like(self.weight_mu)
            b = self.bias_mu + self._sigma(self.bias_rho) * torch.randn_like(self.bias_mu)
        elif sample:  # eval-time posterior sample
            w = self.weight_mu + self._sigma(self.weight_rho) * torch.randn_like(self.weight_mu)
            b = self.bias_mu + self._sigma(self.bias_rho) * torch.randn_like(self.bias_mu)
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        # KL( N(mu, sigma^2) || N(0, prior_sigma^2) ) summed over all params
        ps2 = self.prior_sigma ** 2
        kl = 0.0
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sig2 = self._sigma(rho) ** 2
            kl = kl + 0.5 * (sig2.sum() / ps2 + (mu ** 2).sum() / ps2
                             - mu.numel() + mu.numel() * math.log(ps2) - torch.log(sig2).sum())
        return kl


def _scatter_mean(messages, dst, n_nodes):
    """Mean of `messages` grouped by destination node (native index_add_)."""
    out = torch.zeros(n_nodes, messages.shape[1], device=messages.device, dtype=messages.dtype)
    out.index_add_(0, dst, messages)
    deg = torch.zeros(n_nodes, device=messages.device, dtype=messages.dtype)
    deg.index_add_(0, dst, torch.ones(dst.shape[0], device=messages.device, dtype=messages.dtype))
    return out / deg.clamp_(min=1.0).unsqueeze(1)


class HeteroSAGELayer(nn.Module):
    """h' = act( W_self h + Σ_r W_r mean_{u∈N_r(v)} h_u )."""

    def __init__(self, in_dim, out_dim, edge_types, prior_sigma=1.0):
        super().__init__()
        self.edge_types = edge_types
        self.self_lin = BayesianLinear(in_dim, out_dim, prior_sigma)
        self.rel_lins = nn.ModuleList(
            [BayesianLinear(in_dim, out_dim, prior_sigma) for _ in edge_types])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h, rel_index, sample=True, active_rel=None):
        n = h.shape[0]
        out = self.self_lin(h, sample)
        for r, (src, dst) in enumerate(rel_index):
            if active_rel is not None and r not in active_rel:
                continue
            if src.numel() == 0:
                continue
            agg = _scatter_mean(h[src], dst, n)
            out = out + self.rel_lins[r](agg, sample)
        return self.norm(F.relu(out))

    def kl(self):
        return self.self_lin.kl() + sum(l.kl() for l in self.rel_lins)


class SoilGNN(nn.Module):
    def __init__(self, *, cat_cardinalities, num_dim, n_codes, n_families, n_drains,
                 edge_types, hidden=128, layers=3, dropout=0.2, prior_sigma=1.0,
                 emb_dim_cap=16, n_uscs=0):
        super().__init__()
        self.edge_types = edge_types
        self.n_codes, self.n_families, self.n_drains = n_codes, n_families, n_drains
        self.n_uscs = n_uscs
        # per-column embeddings (dim ~ sqrt(cardinality), capped)
        self.embs = nn.ModuleList()
        emb_total = 0
        for card in cat_cardinalities:
            d = min(emb_dim_cap, max(2, int(round(card ** 0.5))))
            self.embs.append(nn.Embedding(card, d))
            emb_total += d
        in_dim = emb_total + num_dim
        self.input = BayesianLinear(in_dim, hidden, prior_sigma)
        self.convs = nn.ModuleList(
            [HeteroSAGELayer(hidden, hidden, edge_types, prior_sigma) for _ in range(layers)])
        self.dropout = dropout
        # heads consume [graph-context h_final ; per-node input projection h0] (jumping-knowledge
        # skip) so the classifier keeps direct access to the node's own geology signal — without
        # it the 3 SAGE layers oversmooth and a plain RF on the same features wins.
        head_in = hidden * 2
        self.family_head = BayesianLinear(head_in, n_families, prior_sigma)
        self.code_head = BayesianLinear(head_in + n_families, n_codes, prior_sigma)
        self.drain_head = BayesianLinear(head_in, n_drains, prior_sigma)
        # auxiliary near-surface USCS head (from OCR'd borings) — shares the encoder
        self.uscs_head = BayesianLinear(head_in, n_uscs, prior_sigma) if n_uscs > 0 else None

    def encode(self, x_num, x_mask, cat_idx, rel_index, sample=True, active_rel=None):
        parts = [emb(cat_idx[:, j]) for j, emb in enumerate(self.embs)]
        parts.append(x_num)
        parts.append(x_mask)
        h0 = F.relu(self.input(torch.cat(parts, dim=1), sample))
        h = h0
        for conv in self.convs:
            h = conv(h, rel_index, sample, active_rel)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return torch.cat([h, h0], dim=1)   # jumping-knowledge skip

    def heads(self, h, sample=True, prior_logits=None):
        fam_logits = self.family_head(h, sample)
        code_logits = self.code_head(torch.cat([h, fam_logits], dim=1), sample)
        if prior_logits is not None:
            code_logits = code_logits + prior_logits
        drain_logits = self.drain_head(h, sample)
        uscs_logits = self.uscs_head(h, sample) if self.uscs_head is not None else None
        return fam_logits, code_logits, drain_logits, uscs_logits

    def forward(self, x_num, x_mask, cat_idx, rel_index, sample=True,
                prior_logits=None, active_rel=None):
        h = self.encode(x_num, x_mask, cat_idx, rel_index, sample, active_rel)
        return self.heads(h, sample, prior_logits)

    def kl(self):
        k = self.input.kl() + self.family_head.kl() + self.code_head.kl() + self.drain_head.kl()
        if self.uscs_head is not None:
            k = k + self.uscs_head.kl()
        for conv in self.convs:
            k = k + conv.kl()
        return k


def build_rel_index(edge_index, edge_type, n_edge_types, device):
    """Precompute (src, dst) index tensors per relation, once."""
    rel = []
    for r in range(n_edge_types):
        m = edge_type == r
        rel.append((edge_index[0, m].to(device), edge_index[1, m].to(device)))
    return rel


def fourier_depth(z, n_freq=6):
    """Fourier features of a standardized depth scalar: [z, sin(2^k z), cos(2^k z)]."""
    feats = [z]
    for k in range(n_freq):
        f = 2.0 ** k
        feats.append(torch.sin(f * z))
        feats.append(torch.cos(f * z))
    return torch.cat(feats, dim=-1)


class SoilGNN3D(nn.Module):
    """3D depth-resolved model: the GraphSAGE encoder produces a per-boring spatial latent;
    a depth-conditioned decoder predicts SPT-N (heteroscedastic), USCS class, and groundwater
    as a function of (latent, depth). Reuses the Phase-A encoder verbatim; only the heads change.
    """

    def __init__(self, *, cat_cardinalities, num_dim, edge_types, n_uscs, n_freq=6,
                 hidden=128, layers=3, dropout=0.2, prior_sigma=1.0, emb_dim_cap=16):
        super().__init__()
        self.edge_types = edge_types
        self.n_freq = n_freq
        self.embs = nn.ModuleList()
        emb_total = 0
        for card in cat_cardinalities:
            d = min(emb_dim_cap, max(2, int(round(card ** 0.5))))
            self.embs.append(nn.Embedding(card, d))
            emb_total += d
        self.input = BayesianLinear(emb_total + num_dim, hidden, prior_sigma)
        self.convs = nn.ModuleList(
            [HeteroSAGELayer(hidden, hidden, edge_types, prior_sigma) for _ in range(layers)])
        self.dropout = dropout
        depth_dim = 1 + 2 * n_freq
        dec_in = hidden * 2 + depth_dim          # [JK latent ; γ(depth)]
        self.spt_head = BayesianLinear(dec_in, 2, prior_sigma)     # μ, logσ² (log1p N space)
        self.uscs_head = BayesianLinear(dec_in, n_uscs, prior_sigma)
        self.gw_head = BayesianLinear(hidden * 2, 2, prior_sigma)  # per-boring, depth-independent

    def encode(self, x_num, x_mask, cat_idx, rel_index, sample=True, active_rel=None):
        parts = [emb(cat_idx[:, j]) for j, emb in enumerate(self.embs)]
        parts += [x_num, x_mask]
        h0 = F.relu(self.input(torch.cat(parts, dim=1), sample))
        h = h0
        for conv in self.convs:
            h = conv(h, rel_index, sample, active_rel)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return torch.cat([h, h0], dim=1)

    def decode(self, h_sel, depth_std, sample=True):
        """h_sel [M, 2*hidden] latents for the sampled nodes; depth_std [M,1] standardized depth."""
        g = fourier_depth(depth_std, self.n_freq)
        z = torch.cat([h_sel, g], dim=1)
        spt = self.spt_head(z, sample)            # [M,2]
        uscs = self.uscs_head(z, sample)          # [M,n_uscs]
        return spt, uscs

    def gw(self, h_sel, sample=True):
        return self.gw_head(h_sel, sample)        # [M,2] per boring

    def kl(self):
        k = self.input.kl() + self.spt_head.kl() + self.uscs_head.kl() + self.gw_head.kl()
        for conv in self.convs:
            k = k + conv.kl()
        return k
