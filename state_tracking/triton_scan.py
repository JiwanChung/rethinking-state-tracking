import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _combine(a_l, b_l, a_r, b_r):
    return a_l * a_r, a_r * b_l + b_r


@triton.jit
def _scan_fwd(gates_ptr, values_ptr, out_ptr, stride_n, stride_t,
              T: tl.constexpr, BLOCK_T: tl.constexpr):
    n_idx = tl.program_id(0)
    d_idx = tl.program_id(1)
    t = tl.arange(0, BLOCK_T)
    mask = t < T
    ptrs = n_idx * stride_n + t * stride_t + d_idx
    a = tl.load(gates_ptr + ptrs, mask=mask, other=1.0)
    b = tl.load(values_ptr + ptrs, mask=mask, other=0.0)
    _, h = tl.associative_scan((a, b), 0, _combine)
    tl.store(out_ptr + ptrs, h, mask=mask)


def _run_scan(gates, values):
    values = values.to(gates.dtype)
    N, T, D = gates.shape
    g = gates.transpose(1, 2).contiguous().view(N * D, T, 1)
    v = values.transpose(1, 2).contiguous().view(N * D, T, 1)
    out = torch.empty_like(v)
    _scan_fwd[(N * D, 1)](g, v, out, g.stride(0), g.stride(1),
                          T=T, BLOCK_T=triton.next_power_of_2(T))
    return out.view(N, D, T).transpose(1, 2).contiguous()


class TritonScan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gates, values):
        shape = gates.shape
        T, D = shape[-2], shape[-1]
        N = gates.numel() // (T * D)
        h = _run_scan(gates.reshape(N, T, D), values.reshape(N, T, D)).view(shape)
        ctx.save_for_backward(gates, h)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        gates, h = ctx.saved_tensors
        shape = gates.shape
        T, D = shape[-2], shape[-1]
        N = gates.numel() // (T * D)
        gates_shift = F.pad(gates[..., 1:, :], (0, 0, 0, 1), value=0.0)
        g = _run_scan(gates_shift.flip(-2).contiguous().reshape(N, T, D),
                      grad_h.flip(-2).contiguous().reshape(N, T, D))
        g = g.view(shape).flip(-2).contiguous()
        h_prev = F.pad(h[..., :-1, :], (0, 0, 1, 0), value=0.0)
        return g * h_prev, g
