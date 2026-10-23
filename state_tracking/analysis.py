import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from state_tracking.data import StateTrackingDataset
from state_tracking.groups import GROUPS
from state_tracking.models import (AUSSMLayer, GatedRNNLayer, LinearRNNLayer,
                                   Mamba3Layer, NegativeMambaLayer, SimpleAUSSMLayer)


@torch.no_grad()
def rollouts(model, group_name, n_seqs, seq_len, seed, device, batch_size=4):
    """Hidden states fed to the readout, (n_seqs, seq_len, d_model), and labels."""
    model.eval()
    ds = StateTrackingDataset(GROUPS[group_name](), n_seqs, seq_len, seed=seed)
    inputs = ds.inputs.to(device)
    captured = []
    handle = model.output_proj.register_forward_hook(
        lambda _m, inp, _out: captured.append(inp[0].detach().to(torch.float32).cpu()))
    try:
        for start in range(0, n_seqs, batch_size):
            model(inputs[start:start + batch_size])
    finally:
        handle.remove()
    return torch.cat(captured, dim=0).numpy(), ds.labels.numpy()


def distinguishability(h, labels, model=None):
    """R(t), M(t), q(t) per step.

    With `model`, states are mapped to readout space z = W_out h + b; without it,
    the latent states h are used.
        c_g(t) = mean of z over rollouts with label g at step t
        R(t)   = mean over rollouts of ||z_i(t) - c_{g_i(t)}(t)||
        M(t)   = min over g != g' of ||c_g(t) - c_g'(t)||
    Returns arrays of length T: R, R_q25, R_q75, M, q, n_finite (the number of
    rollouts with a finite state at step t; the others are left out).
    """
    if model is not None:
        W = model.output_proj.weight.detach().cpu().double().numpy()
        b = model.output_proj.bias.detach().cpu().double().numpy()
    N, T, _ = h.shape
    out = {k: np.full(T, np.nan) for k in ("R", "R_q25", "R_q75", "M")}
    out["n_finite"] = np.zeros(T, dtype=np.int64)
    for t in range(T):
        z = h[:, t].astype(np.float64)
        if model is not None:
            z = z @ W.T + b
        fin = np.isfinite(z).all(axis=-1)
        out["n_finite"][t] = int(fin.sum())
        groups = [g for g in np.unique(labels[fin, t])]
        if len(groups) < 2:
            continue
        c = {g: z[fin & (labels[:, t] == g)].mean(axis=0) for g in groups}
        out["M"][t] = min(np.linalg.norm(c[g] - c[k])
                          for i, g in enumerate(groups) for k in groups[i + 1:])
        norms = np.linalg.norm(z[fin] - np.stack([c[g] for g in labels[fin, t]]), axis=-1)
        out["R"][t] = norms.mean()
        out["R_q25"][t], out["R_q75"][t] = np.percentile(norms, [25, 75])
    out["q"] = out["R"] / out["M"]
    return out


def subspace_split(h, labels):
    """Per step: U(t) is spanned by the centered class centroids (the top right
    singular vectors with singular value > 1e-6 of the largest). With
    delta_i = h_i - c_{g_i}(t):
        r_U(t)    = root mean square of ||P_U delta_i||
        r_perp(t) = root mean square of ||delta_i - P_U delta_i||
        r_sep(t)  = min over g != g' of ||c_g(t) - c_g'(t)||
    Returns arrays of length T: r_U, r_perp, r_sep, q_U = r_U / r_sep,
    q_perp = r_perp / r_sep, n_finite (rollouts with a finite state at step t).
    """
    N, T, D = h.shape
    out = {k: np.full(T, np.nan) for k in ("r_U", "r_perp", "r_sep")}
    out["n_finite"] = np.zeros(T, dtype=np.int64)
    for t in range(T):
        x = h[:, t].astype(np.float64)
        fin = np.isfinite(x).all(axis=-1)
        out["n_finite"][t] = int(fin.sum())
        groups = [g for g in np.unique(labels[fin, t])]
        if len(groups) < 2:
            continue
        c = np.stack([x[fin & (labels[:, t] == g)].mean(axis=0) for g in groups])
        _, s, vt = np.linalg.svd(c - c.mean(axis=0), full_matrices=False)
        basis = vt[: int((s > s[0] * 1e-6).sum())].T
        index = {g: i for i, g in enumerate(groups)}
        delta = x[fin] - c[[index[g] for g in labels[fin, t]]]
        sq_U = ((delta @ basis) ** 2).sum(axis=-1)
        sq = (delta ** 2).sum(axis=-1)
        out["r_U"][t] = np.sqrt(sq_U.mean())
        out["r_perp"][t] = np.sqrt(np.maximum(sq - sq_U, 0.0).mean())
        out["r_sep"][t] = min(np.linalg.norm(c[i] - c[j])
                              for i in range(len(groups)) for j in range(i + 1, len(groups)))
    out["q_U"] = out["r_U"] / out["r_sep"]
    out["q_perp"] = out["r_perp"] / out["r_sep"]
    return out


def _recurrent_layer(model):
    block = model.blocks[0]
    for attr in ("rnn", "layer", "mamba"):
        if hasattr(block, attr):
            return getattr(block, attr)
    raise ValueError(f"block {type(block).__name__} has no recurrent layer")


def _step_fn(layer, x, t):
    x_t = x[:, t]
    if isinstance(layer, nn.RNN):
        inp = F.linear(x_t, layer.weight_ih_l0, layer.bias_ih_l0)
        return layer.hidden_size, lambda h: torch.tanh(inp + F.linear(h, layer.weight_hh_l0, layer.bias_hh_l0))

    if isinstance(layer, LinearRNNLayer):
        inp = layer.W_ih(x_t)
        return layer.d_model, lambda h: inp + layer.W_hh(h)

    if isinstance(layer, GatedRNNLayer):
        inp = layer.W_ih(x_t)
        gx = layer.gate_x(x_t)
        if layer.gate_use_h:
            return layer.d_model, lambda h: torch.sigmoid(gx + layer.gate_h(h)) * layer.W_hh(h) + inp
        g = torch.sigmoid(gx)
        return layer.d_model, lambda h: g * layer.W_hh(h) + inp

    if isinstance(layer, NegativeMambaLayer):
        delta = F.softplus(layer.delta_proj(x_t))
        dA = 2 * torch.exp(delta * -torch.exp(layer.A_log.to(x_t.dtype))) - 1
        val = delta * layer.B_proj(x_t)
        return layer.d_state, lambda h: dA * h + val

    if isinstance(layer, Mamba3Layer):
        H, P, N = layer.nheads, layer.headdim, layer.d_state
        z, xs, B_in, C_in, dt_raw, lam_raw, theta = torch.split(
            layer.in_proj(x_t), list(layer._proj_sizes), dim=-1)
        dt = F.softplus(dt_raw + layer.dt_bias)
        gamma = torch.sigmoid(lam_raw) * dt
        alpha = torch.exp(dt * -torch.exp(layer.A_log.to(x_t.dtype)))
        B_t = layer._apply_rope(layer.B_norm(B_in).unsqueeze(-2) + layer.B_bias,
                                -(dt.unsqueeze(-1) * theta.unsqueeze(-2)))
        xs = xs.view(*xs.shape[:-1], H, P)
        forcing = (gamma[..., None, None] * xs[..., :, None] * B_t[..., None, :]).reshape(*xs.shape[:-2], H * P * N)
        a = alpha[..., None, None].expand(*alpha.shape, P, N).reshape(*alpha.shape[:-1], H * P * N)
        return H * P * N, lambda h: a * h + forcing

    if type(layer).__name__ == "Mamba":
        d_inner, N, R = layer.d_inner, layer.d_state, layer.dt_rank
        K = layer.conv1d.weight.shape[-1]
        x_in, _ = layer.in_proj(x[:, max(0, t - K + 1):t + 1]).chunk(2, dim=-1)
        taps = layer.conv1d.weight[:, 0, K - x_in.shape[1]:]
        x_inner = F.silu((x_in * taps.T).sum(dim=1) + layer.conv1d.bias)
        dt_pre, B_t, _ = torch.split(layer.x_proj(x_inner), [R, N, N], dim=-1)
        dt = F.softplus(layer.dt_proj(dt_pre))
        A_bar = torch.exp(dt.unsqueeze(-1) * -torch.exp(layer.A_log.float()))
        bx = dt.unsqueeze(-1) * B_t.unsqueeze(-2) * x_inner.unsqueeze(-1)
        a = A_bar.reshape(*A_bar.shape[:-2], d_inner * N)
        b = bx.reshape(*bx.shape[:-2], d_inner * N)
        return d_inner * N, lambda h: a * h + b

    if isinstance(layer, (AUSSMLayer, SimpleAUSSMLayer)):
        raise NotImplementedError(
            "perturbation_recovery has no one-step map for AUSSM / Simple AUSSM")
    raise ValueError(f"no one-step map for layer {type(layer).__name__}")


@torch.no_grad()
def perturbation_recovery(model, group_name, device, eval_len=200, n_seqs=200,
                          sigma=1e-2, inject_step=20, seed=12345):
    """Run clean and perturbed copies of the first layer's recurrence on the same
    inputs; at `inject_step`, add N(0, sigma^2) noise to the perturbed state.
    Returns ||e_t|| / ||e_t0|| for t = t0 .. eval_len-1, shape (n_seqs, eval_len - t0).
    """
    model.eval()
    ds = StateTrackingDataset(GROUPS[group_name](), n_seqs, eval_len, seed=seed)
    x_norm = model.blocks[0].norm(model.embedding(ds.inputs.to(device)))
    layer = _recurrent_layer(model)
    d, _ = _step_fn(layer, x_norm, 0)
    h_clean = torch.zeros(n_seqs, d, device=device)
    h_pert = torch.zeros(n_seqs, d, device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + 1)
    errors = []
    for t in range(eval_len):
        _, step = _step_fn(layer, x_norm, t)
        h_clean, h_pert = step(h_clean), step(h_pert)
        if t == inject_step:
            h_pert = h_pert + torch.randn(n_seqs, d, device=device, generator=rng) * sigma
        if t >= inject_step:
            errors.append((h_pert - h_clean).norm(dim=-1).cpu().numpy())
    errors = np.stack(errors, axis=1)
    return errors / errors[:, :1]
