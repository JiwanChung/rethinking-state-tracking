"""Parallel scan for the element-wise linear recurrence h[t] = a[t] * h[t-1] + b[t].

On a GPU the scan runs as one Triton kernel (triton_scan.py), as in the runs behind
the paper; on a CPU it runs as a Hillis-Steele scan in PyTorch. Both compute the
same recurrence, but their floating-point rounding differs, and over a long training
run that can change its outcome. Both backward passes are a reverse scan, so only
the gates and the output are kept.
"""

import torch
import torch.nn.functional as F


def parallel_scan(gates, values):
    """gates, values: (*, T, D) -> all hidden states (*, T, D)."""
    if gates.is_cuda:
        from state_tracking.triton_scan import TritonScan   # Triton ships with CUDA builds of torch
        return TritonScan.apply(gates.contiguous(), values.contiguous())
    return _TorchScan.apply(gates, values)


def _torch_scan(gates, values):
    a, b = gates.clone(), values.clone()
    T = a.shape[-2]
    stride = 1
    while stride < T:
        a_prev = F.pad(a[..., :-stride, :], (0, 0, stride, 0), value=1.0)
        b_prev = F.pad(b[..., :-stride, :], (0, 0, stride, 0), value=0.0)
        b = a * b_prev + b
        a = a * a_prev
        stride *= 2
    return b


class _TorchScan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gates, values):
        with torch.no_grad():
            h = _torch_scan(gates, values)
        ctx.save_for_backward(gates, h)
        h = h.clone()
        h.requires_grad_(gates.requires_grad or values.requires_grad)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        gates, h = ctx.saved_tensors
        with torch.no_grad():
            # g[t] = grad_h[t] + gates[t+1] * g[t+1]: a reverse scan.
            gates_shifted = F.pad(gates[..., 1:, :], (0, 0, 0, 1), value=0.0)
            g = _torch_scan(gates_shifted.flip(-2), grad_h.flip(-2)).flip(-2)
            h_prev = F.pad(h[..., :-1, :], (0, 0, 1, 0), value=0.0)
        return g * h_prev, g
