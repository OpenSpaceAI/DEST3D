import math
import torch
import triton
import triton.language as tl
from einops import rearrange

from issm_triton.issm_cabt_bmm import _bmm_CABT_chunk_fwd, _bmm_CABT_dbc_chunk_bwd
from issm_triton.issm_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd
from issm_triton.issm_chunk_state import _chunk_state_fwd, _chunk_state_bwd_db
from issm_triton.issm_state_passing import _state_passing_fwd, _state_passing_bwd
from issm_triton.issm_chunk_scan import _chunk_scan_fwd, _chunk_scan_bwd_dz, _chunk_scan_bwd_dstates
from issm_triton.issm_chunk_scan import _chunk_scan_bwd_dc, _chunk_scan_bwd_dcb
from issm_triton.issm_chunk_scan import _chunk_scan_bwd_ddAcs_stable


def ISSM_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads, dstate)
        A: (nheads, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    return ISSMChunkScanCombinedFn.apply(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, dt_softplus, dt_limit, return_final_states)


class ISSMChunkScanCombinedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False):
        ctx.dt_dtype = dt.dtype
        out, out_x, dt_out, dA_cumsum, states, final_states = _issm_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=dt_softplus, dt_limit=dt_limit)
        ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx)
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        return out if not return_final_states else (out, final_states)

    @staticmethod
    def backward(ctx, dout, *args):
        out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _issm_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit)
        return dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, dinitial_states, None, None, None, None


def _issm_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads, dstate)
    assert A.shape == (nheads, dstate)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, final_states = _state_passing_fwd(states, dA_cumsum[:, :, :, -1, :], initial_states=initial_states, seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
    dA_cumsum_bmm = rearrange(dA_cumsum, "b h nc cs d -> b h (nc cs) d").contiguous()
    dt_bmm = rearrange(dt, "b h nc cs d -> b h (nc cs) d").contiguous()
    CABT = _bmm_CABT_chunk_fwd(B, C, dA_cumsum_bmm, dt_bmm, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _chunk_scan_fwd(CABT, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)
    return out, out_x, dt, dA_cumsum, states, final_states


def _issm_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D=None, z=None,
                                   dt_bias=None, initial_states=None, dfinal_states=None, seq_idx=None, dt_softplus=False,
                                   dt_limit=(0.0, float("inf")),
                                   dx=None, ddt=None, dB=None, dC=None, dz=None, recompute_output=False):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads, dstate)
    assert A.shape == (nheads, dstate)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)
    dt_in = dt.clone()
    dA_cumsum, dt = _chunk_cumsum_fwd(dt_in, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)

    dA_cumsum_bmm = rearrange(dA_cumsum, "b h nc cs d -> b h (nc cs) d").contiguous()
    dt_bmm = rearrange(dt, "b h nc cs d -> b h (nc cs) d").contiguous()
    CABT = _bmm_CABT_chunk_fwd(B, C, dA_cumsum_bmm, dt_bmm, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)

    states, _ = _state_passing_fwd(states, dA_cumsum[:, :, :, -1, :], initial_states=initial_states, seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, has_ddAcs=False, D=D, dz=dz, recompute_output=recompute_output)
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out
    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype)
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        states, dA_cumsum[:, :, :, -1, :], dstates, dfinal_states=dfinal_states, seq_idx=seq_idx,
        has_initial_states=initial_states is not None,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    ddA_chunk_cumsum = ddA_chunk_cumsum.sum(dim=-2)

    dx, dD_from_x = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CABT, dout, dstates, D=D, seq_idx=seq_idx, dx=dx)
    dB, ddA_next = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups)
    dC, ddA_cumsum_prev = _chunk_scan_bwd_dc(states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups)

    dCABT = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups).to(CABT.dtype)
    ddt = _chunk_scan_chunk_state_bwd_dt(x, dt, dA_cumsum, B, C, dout, dstates, dCABT, seq_idx=seq_idx)
    _bmm_CABT_dbc_chunk_bwd(B, C, dt_bmm, dA_cumsum_bmm, dCABT, dB=dB, dC=dC, dB_out=dB_given, dC_out=dC_given)

    if z is None:
        dD = dD_from_x

    ddA_cumsum_prev[..., -1, :] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-2]).cumsum(dim=-2).flip([-2])
    ddA = _chunk_scan_bwd_ddAcs_stable(dt, dA_cumsum, B, C, dCABT)
    ddA += ddA_next + ddA_prev
    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt, dt_in, A, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit, ddt=ddt_given)

    return_vals = (dx, ddt_given, dA, dB_given, dC_given, dD, dz, ddt_bias, dinitial_states)
    return return_vals if not recompute_output else (*return_vals, outz)


# *chunk scan & state backward dx*
def _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size, _ = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, nheads, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size, dstate)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    # ddt = torch.empty(batch, nheads, nchunks, chunk_size, dstate, device=dout.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                        batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x, CB, dout, dt, dA_cumsum, seq_idx, D, B, dstates, dx, dD,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(-1), CB.stride(-2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3), dt.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            D.stride(0) if D is not None else 0,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            # ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3), ddt.stride(4),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return dx, dD

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, D_ptr,
    b_ptr, dstates_ptr,
    dx_ptr, dD_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize, stride_dt_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_dstate,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_D_head,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    # stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize, stride_ddt_dstate,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_h * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    # ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate)
    if BLOCK_SIZE_DSTATE <= 128:
        dt = tl.load(dt_ptr + offs_m[:, None] * stride_dt_csize + offs_dstate[None, :] * stride_dt_dstate, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_m = tl.load(dA_cumsum_ptr + offs_m[:, None] * stride_dA_cs_csize + offs_dstate[None, :] * stride_dA_cs_dstate, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize + offs_dstate[None, :] * stride_dA_cs_dstate, mask=(offs_dstate[None, :] < dstate), other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_last - dA_cs_m)
        else:
            seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
            seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
        
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b * scale * dt, dstates)
    else:
        dt_ptrs = dt_ptr + offs_dstate[None, :] * stride_dt_dstate
        dA_cumsum_ptrs = dA_cumsum_ptr + offs_dstate[None, :] * stride_dA_cs_dstate
        for k in range(0, dstate, BLOCK_SIZE_K):
            dt = tl.load(dt_ptrs + offs_m[:, None] * stride_dt_csize, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0).to(tl.float32)
            dA_cs_m = tl.load(dA_cumsum_ptrs + offs_m[:, None] * stride_dA_cs_csize, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0).to(tl.float32)
            dA_cs_last = tl.load(dA_cumsum_ptrs + (chunk_size - 1) * stride_dA_cs_csize, mask=(offs_dstate[None, :] < dstate - k), other=0.0).to(tl.float32)
            if not HAS_SEQ_IDX:
                scale = tl.exp(dA_cs_last - dA_cs_m)
            else:
                seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
                seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
                scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)

            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b * scale * dt, dstates)
            dt_ptrs += BLOCK_SIZE_K * stride_dt_dstate
            dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_dstate
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        # For some reason setting mask to (offs_m[:, None] < chunk_size_limit) is much slower
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim), other=0.0)
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dx = acc 
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)


def _chunk_scan_chunk_state_bwd_dt(x, dt, dA_cumsum, B, C, dout, dstates, dCABT, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size, _ = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size, dstate)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dCABT.shape == (batch, nchunks, nheads, chunk_size, chunk_size)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, dstate, device=dout.device, dtype=torch.float32)
    grid_dt = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * dstate, batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dt_kernel[grid_dt](
            x, dout, dt, dA_cumsum, seq_idx, B, C, dstates, dCABT, ddt,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3), dt.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), dA_cumsum.stride(4),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dCABT.stride(0), dCABT.stride(1), dCABT.stride(2), dCABT.stride(3), dCABT.stride(4),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3), ddt.stride(4),
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_HDIM=max(triton.next_power_of_2(headdim), 16),
        )
    return ddt

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dt_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, b_ptr, c_ptr, dstates_ptr, dCABT_ptr,
    ddt_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize, stride_dt_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_dA_cs_dstate,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    stride_dCABT_batch, stride_dCABT_chunk, stride_dCABT_head, stride_dCABT_csize_m, stride_dCABT_csize_n,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize, stride_ddt_dstate,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_HDIM: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = dstate
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_d = tl.program_id(axis=0) % num_pid_n
    
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    c_ptr += pid_b * stride_c_batch + pid_c * chunk_size * stride_c_seqlen + (pid_h // nheads_ngroups_ratio) * stride_c_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    dCABT_ptr += pid_b * stride_dCABT_batch + pid_c * stride_dCABT_chunk + pid_h * stride_dCABT_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dstate = pid_d
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    # Might be faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    # However, we're getting error with the Triton compiler 2.1.0 for that code path:
    # Unexpected mma -> mma layout conversion
    # Triton 2.2.0 fixes this
    b = tl.load(b_ptr + (offs_m * stride_b_seqlen + offs_dstate * stride_b_dstate), mask=(offs_m < chunk_size_limit), other=0.0)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize + offs_dstate * stride_dA_cs_dstate, mask=(offs_m < chunk_size_limit), other=0.0).to(tl.float32)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize + offs_dstate * stride_dA_cs_dstate).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(dA_cs_last - dA_cs_m)
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
    
    offs_hdim = tl.arange(0, BLOCK_SIZE_HDIM if BLOCK_SIZE_HDIM <= 128 else BLOCK_SIZE_K)
    if BLOCK_SIZE_HDIM <= 128:
        dstates_ptrs = dstates_ptr + (offs_hdim * stride_dstates_hdim + offs_dstate * stride_dstates_dstate)
        dstates = tl.load(dstates_ptrs, mask=(offs_hdim < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_hdim[None, :] * stride_x_hdim)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_hdim[None, :] < hdim), other=0.0).to(tl.float32)
        acc = tl.sum(x * dstates[None, :], axis=1) * scale * b
    else:
        dstates_ptrs = dstates_ptr + offs_hdim * stride_dstates_hdim
        x_ptrs = x_ptr + offs_hdim[None, :] * stride_x_hdim
        for k in range(0, hdim, BLOCK_SIZE_K):
            dstates = tl.load(dstates_ptrs + offs_dstate * stride_dstates_dstate, mask=(offs_hdim < hdim), other=0.0)
            x = tl.load(x_ptrs + offs_m[:, None] * stride_x_seqlen, mask=(offs_m[:, None] < chunk_size_limit) & (offs_hdim[None, :] < hdim), other=0.0).to(tl.float32)
            acc += tl.sum(x * dstates[None, :], axis=1) * scale * b
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_hdim
            x_ptrs += BLOCK_SIZE_K * stride_x_hdim

    offs_cs = tl.arange(0, BLOCK_SIZE_K)
    dCABT_ptrs = dCABT_ptr + (offs_m[:, None] * stride_dCABT_csize_n + offs_cs[None, :] * stride_dCABT_csize_m)
    b_ptrs = b_ptr + (offs_m * stride_b_seqlen + offs_dstate * stride_b_dstate)
    c_ptrs = c_ptr + (offs_cs * stride_c_seqlen + offs_dstate * stride_c_dstate)
    dAm_ptrs = dA_cumsum_ptr + (offs_cs * stride_dA_cs_csize + offs_dstate * stride_dA_cs_dstate)
    dAk_ptrs = dA_cumsum_ptr + (offs_m * stride_dA_cs_csize + offs_dstate * stride_dA_cs_dstate)
    
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_K)):
        mask = (offs_m[:, None] < chunk_size_limit) & \
               (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_K) & \
               (offs_m[:, None] <= (offs_cs[None, :] + BLOCK_SIZE_K * cs))
        
        dCABT = tl.load(dCABT_ptrs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_m < chunk_size_limit), other=0.0).to(tl.float32)
        c = tl.load(c_ptrs, mask=(offs_cs < chunk_size_limit - cs * BLOCK_SIZE_K), other=0.0).to(tl.float32)
        dAm = tl.load(dAm_ptrs, mask=(offs_cs < chunk_size_limit - cs * BLOCK_SIZE_K), other=0.0).to(tl.float32)
        dAk = tl.load(dAk_ptrs, mask=(offs_m < chunk_size_limit), other=0.0).to(tl.float32)
        
        temp = tl.where(mask, dCABT * c[None, :] * b[:, None] * tl.exp(dAm[None, :] - dAk[:, None]), 0.0)
        acc += tl.sum(temp, axis=1)

        dCABT_ptrs += BLOCK_SIZE_K * stride_dCABT_csize_m
        c_ptrs += BLOCK_SIZE_K * stride_c_seqlen
        dAm_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize + offs_dstate * stride_ddt_dstate
    tl.store(ddt_ptrs, acc, mask=(offs_m < chunk_size_limit))