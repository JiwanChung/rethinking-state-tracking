import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from state_tracking.scan import parallel_scan as _parallel_scan


MIXERS = ("mamba", "mamba3", "aussm", "simple_aussm", "neg_mamba",
          "linear_rnn", "token_gated_rnn", "tanh_rnn", "state_gated_rnn")

LABELS = {
    "mamba": "Mamba", "mamba3": "Mamba-3", "aussm": "AUSSM",
    "simple_aussm": "Simple AUSSM", "neg_mamba": "Negative Mamba",
    "linear_rnn": "Linear RNN", "token_gated_rnn": "Token-gated RNN",
    "tanh_rnn": "tanh RNN", "state_gated_rnn": "State-gated RNN",
}


class StateTrackingModel(nn.Module):
    def __init__(self, group_order, mixer, d_model=698, n_layers=1, d_state=64,
                 dropout=0.0):
        super().__init__()
        if mixer not in MIXERS:
            raise ValueError(f"unknown mixer {mixer!r}; choose one of {MIXERS}")
        self.group_order = group_order
        self.d_model = d_model
        self.mixer_type = mixer

        self.embedding = nn.Embedding(group_order, d_model)
        self.blocks = nn.ModuleList(
            [build_block(mixer, d_model=d_model, d_state=d_state, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(d_model, group_order)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """x: (B, T) group-element indices -> (B, T, |G|) logits."""
        h = self.embedding(x)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i < len(self.blocks) - 1:
                h = F.gelu(h)
        return self.output_proj(h)


def build_block(mixer, d_model, d_state, dropout):
    if mixer == "mamba":
        return MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
    if mixer == "mamba3":
        return Mamba3Block(d_model=d_model, d_state=d_state, dropout=dropout)
    if mixer == "aussm":
        return AUSSMBlock(d_model=d_model, d_state=d_state, dropout=dropout)
    if mixer == "simple_aussm":
        return SimpleAUSSMBlock(d_model=d_model, d_state=d_state, dropout=dropout)
    if mixer == "neg_mamba":
        return NegativeMambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
    if mixer == "linear_rnn":
        return LinearRNNBlock(d_model=d_model, dropout=dropout)
    if mixer == "token_gated_rnn":
        return GatedRNNBlock(d_model=d_model, gate_use_h=False, dropout=dropout)
    if mixer == "tanh_rnn":
        return RNNBlock(d_model=d_model, dropout=dropout)
    if mixer == "state_gated_rnn":
        return GatedRNNBlock(d_model=d_model, gate_use_h=True, dropout=dropout)
    raise ValueError(f"unknown mixer {mixer!r}; choose one of {MIXERS}")


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError(
                "mixer 'mamba' uses the official Mamba layer; install it with "
                "`pip install mamba-ssm causal-conv1d` (needs CUDA)"
            ) from e
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mamba(self.norm(x)))


class NegativeMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer = NegativeMambaLayer(d_model=d_model, d_state=d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.layer(self.norm(x)))

class AUSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer = AUSSMLayer(d_model=d_model, d_state=d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.layer(self.norm(x)))

class SimpleAUSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer = SimpleAUSSMLayer(d_model=d_model, d_state=d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.layer(self.norm(x)))

class Mamba3Block(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer = Mamba3Layer(d_model=d_model, d_state=d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.layer(self.norm(x)))

class RNNBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.rnn = nn.RNN(d_model, d_model, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.rnn(self.norm(x))
        return x + self.drop(h)


class LinearRNNBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer = LinearRNNLayer(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.layer(self.norm(x)))


class GatedRNNBlock(nn.Module):
    def __init__(self, d_model, gate_use_h, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.rnn = GatedRNNLayer(d_model, gate_use_h=gate_use_h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.rnn(self.norm(x)))


class LinearRNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_ih = nn.Linear(d_model, d_model, bias=True)
        self.W_hh = nn.Linear(d_model, d_model, bias=True)
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.d_model)
        for _, p in self.named_parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x):
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            h = self.W_ih(x[:, t]) + self.W_hh(h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class GatedRNNLayer(nn.Module):
    def __init__(self, d_model, gate_use_h):
        super().__init__()
        self.d_model = d_model
        self.gate_use_h = gate_use_h
        self.W_ih = nn.Linear(d_model, d_model, bias=True)
        self.W_hh = nn.Linear(d_model, d_model, bias=True)
        if gate_use_h:
            self.gate_h = nn.Linear(d_model, d_model, bias=False)
        self.gate_x = nn.Linear(d_model, d_model, bias=True)
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.d_model)
        nn.init.uniform_(self.W_ih.weight, -std, std)
        nn.init.uniform_(self.W_ih.bias, -std, std)
        nn.init.uniform_(self.W_hh.weight, -std, std)
        nn.init.uniform_(self.W_hh.bias, -std, std)
        if self.gate_use_h:
            nn.init.normal_(self.gate_h.weight, std=0.01)
        nn.init.normal_(self.gate_x.weight, std=0.01)
        nn.init.constant_(self.gate_x.bias, 1.0)

    def _compute_gate(self, h, x):
        raw = self.gate_x(x)
        if self.gate_use_h:
            raw = raw + self.gate_h(h)
        return torch.sigmoid(raw)

    def forward(self, x):
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        inp = self.W_ih(x)
        for t in range(L):
            g = self._compute_gate(h, x[:, t])
            h = g * self.W_hh(h) + inp[:, t]
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class NegativeMambaLayer(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.delta_proj = nn.Linear(d_model, d_state, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)

        h0_val = 1.0 / math.sqrt(d_state)
        self.register_buffer('h0', torch.full((d_state,), h0_val))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.delta_proj.weight, std=0.02)
        nn.init.zeros_(self.delta_proj.bias)
        for proj in [self.B_proj, self.C_proj, self.out_proj]:
            nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.D_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.D_proj.bias)

    def forward(self, x):
        B, L, _ = x.shape

        delta = F.softplus(self.delta_proj(x))
        A = -torch.exp(self.A_log.to(x.dtype))
        dA = 2 * torch.exp(delta * A) - 1

        B_t = self.B_proj(x)
        val = delta * B_t

        h0 = self.h0.unsqueeze(0).expand(B, -1)
        val = val.clone()
        val[:, 0] = val[:, 0] + dA[:, 0] * h0

        h = _parallel_scan(dA, val)

        C_t = self.C_proj(x)
        readout = C_t * h
        y = F.gelu(self.out_proj(readout) + self.D_proj(x))
        return y

class Mamba3Layer(nn.Module):
    def __init__(self, d_model, d_state=64, headdim=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim if headdim is not None else d_state
        self.nheads = max(1, d_model // self.headdim)
        self.d_inner = self.nheads * self.headdim
        assert d_state % 2 == 0, "d_state must be even for RoPE on pairs"

        proj_out = (
            self.d_inner
            + self.d_inner
            + d_state
            + d_state
            + self.nheads
            + self.nheads
            + d_state // 2
        )
        self.in_proj = nn.Linear(d_model, proj_out, bias=False)
        self._proj_sizes = (
            self.d_inner, self.d_inner, d_state, d_state,
            self.nheads, self.nheads, d_state // 2,
        )

        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.A_log = nn.Parameter(torch.empty(self.nheads))

        self.B_norm = nn.LayerNorm(d_state)
        self.C_norm = nn.LayerNorm(d_state)

        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))

        self.D = nn.Parameter(torch.ones(self.nheads, self.headdim))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        nn.init.uniform_(self.dt_bias, 0.001, 0.1)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    @staticmethod
    def _apply_rope(x, angles):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c = torch.cos(angles)
        s = torch.sin(angles)
        r1 = c * x1 - s * x2
        r2 = s * x1 + c * x2
        out = torch.stack([r1, r2], dim=-1).flatten(start_dim=-2)
        return out

    def forward(self, x):
        B, L, _ = x.shape
        H, P, N = self.nheads, self.headdim, self.d_state

        proj = self.in_proj(x)
        z, xs, B_in, C_in, dt_raw, lam_raw, theta = torch.split(
            proj, list(self._proj_sizes), dim=-1,
        )

        dt = F.softplus(dt_raw + self.dt_bias)
        lam = torch.sigmoid(lam_raw)
        B_in = self.B_norm(B_in)
        C_in = self.C_norm(C_in)
        xs = xs.view(B, L, H, P)

        raw_angles = dt.unsqueeze(-1) * theta.unsqueeze(-2)
        cum_angles = -torch.cumsum(raw_angles, dim=1)

        B_bh = B_in.unsqueeze(-2) + self.B_bias
        C_bh = C_in.unsqueeze(-2) + self.C_bias
        B_rope = self._apply_rope(B_bh, cum_angles)
        C_rope = self._apply_rope(C_bh, cum_angles)

        A = -torch.exp(self.A_log.to(x.dtype))
        dA = dt * A.view(1, 1, H)
        alpha = torch.exp(dA)
        beta = (1.0 - lam) * dt * alpha
        gamma = lam * dt

        h = torch.zeros(B, H, P, N, device=x.device, dtype=x.dtype)
        B_prev = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)
        x_prev = torch.zeros(B, H, P, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            a_t = alpha[:, t].view(B, H, 1, 1)
            b_t = beta[:, t].view(B, H, 1, 1)
            g_t = gamma[:, t].view(B, H, 1, 1)

            B_t = B_rope[:, t]
            C_t = C_rope[:, t]
            x_t = xs[:, t]

            forcing = (
                b_t * x_prev.unsqueeze(-1) * B_prev.unsqueeze(-2)
                + g_t * x_t.unsqueeze(-1) * B_t.unsqueeze(-2)
            )
            h = a_t * h + forcing

            y_t = torch.einsum('bhn,bhpn->bhp', C_t, h)
            y_t = y_t + self.D * x_t
            outputs.append(y_t)

            B_prev = B_t
            x_prev = x_t

        y = torch.stack(outputs, dim=1).reshape(B, L, self.d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)

class AUSSMLayer(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=d_conv, groups=d_model, padding=d_conv - 1, bias=True,
        )

        self.delta_proj = nn.Linear(d_model, d_model, bias=True)

        self.xA_proj = nn.Linear(d_model, d_model * d_state, bias=True)

        self.B = nn.Parameter(torch.randn(d_state, dtype=torch.complex64))
        self.C = nn.Parameter(torch.randn(d_state, dtype=torch.complex64))

        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        self.xA_proj.bias.data.uniform_(-math.pi, math.pi)

    def forward(self, x):
        B, L, D = x.shape
        N = self.d_state

        xz = self.in_proj(x)
        x_ssm, z_gate = xz.chunk(2, dim=-1)

        xc = x_ssm.transpose(1, 2)
        xc = self.conv1d(xc)[..., :L]
        x_ssm = xc.transpose(1, 2)

        dt = F.softplus(self.delta_proj(x_ssm))
        u = F.silu(x_ssm)

        w_scaled = self.xA_proj.weight / math.sqrt(D)
        xA_bias = self.xA_proj.bias

        B_theta = self.B.imag.view(1, 1, N)
        B_r = F.softplus(self.B.real.view(1, 1, N))
        C_theta = self.C.imag.view(1, 1, N)
        C_r = F.softplus(self.C.real.view(1, 1, N))

        G = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        state_re = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        state_im = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            a_t_flat = F.linear(x_ssm[:, t], w_scaled, xA_bias)
            A_t = a_t_flat.view(B, N, D).permute(0, 2, 1).contiguous()

            G = G + A_t
            phase_in = B_theta - G
            scale = (dt[:, t] * u[:, t]).unsqueeze(-1)
            state_re = state_re + B_r * torch.cos(phase_in) * scale
            state_im = state_im + B_r * torch.sin(phase_in) * scale

            phase_out = C_theta + G
            read = C_r * (torch.cos(phase_out) * state_re
                          - torch.sin(phase_out) * state_im)
            y_t = read.sum(dim=-1) + self.D * u[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        y = y * F.silu(z_gate)
        return self.out_proj(y)

class SimpleAUSSMLayer(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=d_conv, groups=d_model, padding=d_conv - 1, bias=True,
        )
        self.xA_proj = nn.Linear(d_model, d_model * d_state, bias=True)
        self.B = nn.Parameter(torch.randn(d_state, dtype=torch.complex64))
        self.C = nn.Parameter(torch.randn(d_state, dtype=torch.complex64))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        self.xA_proj.bias.data.uniform_(-math.pi, math.pi)

    def forward(self, x):
        B, L, D = x.shape
        N = self.d_state

        xz = self.in_proj(x)
        x_ssm, z_gate = xz.chunk(2, dim=-1)

        xc = x_ssm.transpose(1, 2)
        xc = self.conv1d(xc)[..., :L]
        x_ssm = xc.transpose(1, 2)

        w_scaled = self.xA_proj.weight / math.sqrt(D)
        xA_bias = self.xA_proj.bias

        u = F.silu(x_ssm)

        B_theta = self.B.imag.view(1, 1, N)
        B_r = F.softplus(self.B.real.view(1, 1, N))
        C_theta = self.C.imag.view(1, 1, N)
        C_r = F.softplus(self.C.real.view(1, 1, N))

        G = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        state_re = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        state_im = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            a_t_flat = F.linear(x_ssm[:, t], w_scaled, xA_bias)
            A_t = a_t_flat.view(B, N, D).permute(0, 2, 1).contiguous()

            G = G + A_t
            phase_in = B_theta - G
            scale = u[:, t].unsqueeze(-1)
            state_re = state_re + B_r * torch.cos(phase_in) * scale
            state_im = state_im + B_r * torch.sin(phase_in) * scale

            phase_out = C_theta + G
            read = C_r * (torch.cos(phase_out) * state_re
                          - torch.sin(phase_out) * state_im)
            y_t = read.sum(dim=-1) + self.D * u[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z_gate)
        return self.out_proj(y)
