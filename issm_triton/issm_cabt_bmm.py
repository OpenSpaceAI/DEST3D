import math
import torch
import triton
import triton.language as tl

# *bmm CABT forward*
def _bmm_CABT_chunk_fwd(b, c, a, dt, chunk_size, seq_idx=None, causal=False, output_dtype=None):
    """
    Argument:
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        c: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        a: (batch, nhead, nchunks, chunk_size, k) or (batch, nhead, nchunks, chunk_size, ngroups, k)
        dt: (batch, nhead, nchunks, chunk_size, k) or (batch, nhead, nchunks, chunk_size, ngroups, k)
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (batch, nchunks, nhead, chunk_size, chunk_size)
    """
    # Check constraints.
    has_groups = b.dim() == 4
    if not has_groups:
        batch, seqlen, k = b.shape
    else:
        batch, seqlen, ngroups, k = b.shape
        nheads = a.shape[1]
    assert c.shape == b.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    if c.stride(-1) != 1 and c.stride(1) != 1:
        c = c.contiguous()
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if dt.stride(-1) != 1 and dt.stride(1) != 1:
        dt = dt.contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    # Allocates output.
    out_dtype = b.dtype if output_dtype is None else output_dtype
    out = torch.empty((batch, nchunks, nheads, chunk_size, chunk_size), device=b.device, dtype=out_dtype)
    dot_dtype = (tl.bfloat16 if b.dtype == torch.bfloat16 or c.dtype == torch.bfloat16 else
                 (tl.float16 if b.dtype == torch.float16 or c.dtype == torch.float16 else tl.float32))
    grid = lambda META: (chunk_size, batch * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']), nchunks if not has_groups else nchunks * nheads)
    with torch.cuda.device(b.device.index):
        _bmm_CABT_chunk_fwd_kernel[grid](
            b, c, a, dt, out, seq_idx,
            seqlen, chunk_size, k, nheads, nheads // (ngroups if has_groups else 1), 
            b.stride(0), b.stride(1), 0 if not has_groups else b.stride(2), b.stride(-1),
            c.stride(0), c.stride(1), 0 if not has_groups else c.stride(2), c.stride(-1),
            a.stride(0), a.stride(1), a.stride(2), a.stride(-1),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(-1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(-2), out.stride(-1),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            causal,
            dot_dtype,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'K', 'IS_CAUSAL'],
)
@triton.jit
def _bmm_CABT_chunk_fwd_kernel(
    # Pointers to matrices
    b_ptr, c_ptr, a_ptr, dt_ptr, out_ptr, seq_idx_ptr,
    # Matrix dimensions
    seqlen, chunk_size, K, nheads, nheads_ngroups_ratio,
    stride_b_batch, stride_b_seqlen, stride_b_group, stride_bk,
    stride_c_batch, stride_c_seqlen, stride_c_group, stride_ck,
    stride_a_batch, stride_a_head, stride_a_seqlen, stride_ak,
    stride_dt_batch, stride_dt_head, stride_dt_seqlen, stride_dtk,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_outm, stride_outn,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_b = tl.program_id(axis=1) // num_pid_n
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1) % num_pid_n
    pid_cg = tl.program_id(axis=2)
    pid_c = pid_cg // nheads
    pid_h = pid_cg - pid_c * nheads
    pid_g = pid_h // nheads_ngroups_ratio
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1):
            return
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_g * stride_b_group
    c_ptr += pid_b * stride_c_batch + pid_c * chunk_size * stride_c_seqlen + pid_g * stride_c_group
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen + pid_h * stride_dt_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_n[:, None] * stride_b_seqlen + offs_k[None, :] * stride_bk)
    c_ptrs = c_ptr + (offs_m * stride_c_seqlen + offs_k[None, :] * stride_ck)
    am_ptrs = a_ptr + (offs_m * stride_a_seqlen + offs_k[None, :] * stride_ak)
    ak_ptrs = a_ptr + (offs_n[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    dt_ptrs = dt_ptr + (offs_n[:, None] * stride_dt_seqlen + offs_k[None, :] * stride_dtk)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        n_mask = offs_n[:, None] < chunk_size_limit
        m_mask = offs_m < chunk_size_limit
        
        b = tl.load(b_ptrs, mask=n_mask & k_mask, other=0.0).to(dot_dtype)
        c = tl.load(c_ptrs, mask=m_mask & k_mask, other=0.0).to(dot_dtype)
        am = tl.load(am_ptrs, mask=m_mask & k_mask, other=0.0).to(dot_dtype)
        ak = tl.load(ak_ptrs, mask=n_mask & k_mask, other=0.0).to(dot_dtype)
        dt = tl.load(dt_ptrs, mask=n_mask & k_mask, other=0.0).to(dot_dtype)
        
        exp_diff = tl.exp(am - ak)
        acc += tl.sum(exp_diff * b * dt * c, axis=1)
        
        b_ptrs += BLOCK_SIZE_K * stride_bk
        c_ptrs += BLOCK_SIZE_K * stride_ck
        dt_ptrs += BLOCK_SIZE_K * stride_dtk
        am_ptrs += BLOCK_SIZE_K * stride_ak
        ak_ptrs += BLOCK_SIZE_K * stride_ak

    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_n == seq_idx_m, acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m + offs_n * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m < chunk_size) & (offs_n < chunk_size))


# *bmm CABT backward dB*
def _bmm_CABT_db_chunk_bwd(a, dt, dA_cumsum, dout, residual=None, out=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dout: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
        residual: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    Return:
        out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)

    If there was seq_idx in the fwd pass, then dout[i, j] for seq_idx[i] != seq_idx[j] should already be
    zeroed out before calling this function.
    """
    # Check constraints.
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
        nheads = dA_cumsum.shape[1]
    nchunks, chunk_size = dout.shape[1], dout.shape[-1]
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if dout.stride(-1) != 1 and dout.stride(-2) != 1:
        dout = dout.contiguous()
    if residual is not None:
        assert residual.shape == (batch, seqlen, k) if not has_groups else (batch, seqlen, ngroups, k)
        if residual.stride(-1) != 1 and residual.stride(1) != 1:
            residual = residual.contiguous()
    # Allocates output.
    if out is not None:
        assert out.shape == a.shape
        assert out.stride(-1) == 1 or out.stride(1) == 1
    else:
        out = torch.empty_like(a)
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or dout.dtype == torch.bfloat16 else
                 (tl.float16 if a.dtype == torch.float16 or dout.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * k, batch,
                    nchunks if not has_groups else nchunks * ngroups)
    residual_strides = ((residual.stride(0), residual.stride(1), 0 if not has_groups else residual.stride(2),
                         residual.stride(-1))
                        if residual is not None else (0, 0, 0, 0))
    with torch.cuda.device(a.device.index):
        _bmm_CABT_chunk_db_bwd_kernel[grid](
            a, dt, dA_cumsum, dout, out, residual,
            seqlen, chunk_size, k, nheads, nheads // ngroups if has_groups else 1, 
            a.stride(0), a.stride(1), 0 if not has_groups else a.stride(2), a.stride(-1),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
            dout.stride(0), dout.stride(1), 0 if not has_groups else dout.stride(2), dout.stride(-2), dout.stride(-1),
            out.stride(0), out.stride(1), 0 if not has_groups else out.stride(2), out.stride(-1),
            residual_strides[0], residual_strides[1], residual_strides[2], residual_strides[3],
            dot_dtype,
            HAS_RESIDUAL=residual is not None,
        )
    return out

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_CS': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_CS': 64}, num_stages=5, num_warps=4),
    ],
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_CABT_chunk_db_bwd_kernel(
    # Pointers to matrices
    a_ptr, dt_ptr, dA_cumsum_ptr, dout_ptr, db_ptr, res_ptr,
    # Matrix dimensions
    seqlen, chunk_size, K, nheads, nheads_ngroups_ratio,
    stride_a_batch, stride_a_seqlen, stride_a_group, stride_ak,
    stride_dt_batch, stride_dt_head, stride_dt_seqlen, stride_dtk,
    stride_A_cumsum_batch, stride_A_cumsum_head, stride_A_cumsum_seqlen, stride_A_cumsum_k,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_seqlen, stride_db_group, stride_db_k,
    stride_res_batch, stride_res_seqlen, stride_res_group, stride_res_k,
    # Meta-parameters
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_cg = tl.program_id(axis=2)
    pid_c = pid_cg // (nheads // nheads_ngroups_ratio)
    pid_g = pid_cg - pid_c * (nheads // nheads_ngroups_ratio)
    num_pid_n = K
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    acc = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    for i in range(0, nheads_ngroups_ratio):
        pid_h = pid_g * nheads_ngroups_ratio + i

        a_ptr_i = a_ptr + pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_g * stride_a_group
        dA_cumsum_ptr_i = dA_cumsum_ptr + pid_b * stride_A_cumsum_batch + pid_c * chunk_size * stride_A_cumsum_seqlen + pid_h * stride_A_cumsum_head
        dt_ptr_i = dt_ptr + pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen + pid_h * stride_dt_head
        dout_ptr_i = dout_ptr + pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n
        offs_cs = tl.arange(0, BLOCK_SIZE_CS)
        tl.multiple_of(offs_m, 8)
        tl.multiple_of(offs_cs, 8)

        dout_ptrs = dout_ptr_i + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
        a_ptrs = a_ptr_i + (offs_cs * stride_a_seqlen + offs_n * stride_ak)
        dt_ptrs = dt_ptr_i + (offs_m * stride_dt_seqlen + offs_n * stride_dtk)
        dAm_ptrs = dA_cumsum_ptr_i + (offs_m * stride_A_cumsum_seqlen + offs_n * stride_A_cumsum_k)
        dAk_ptrs = dA_cumsum_ptr_i + (offs_cs * stride_A_cumsum_seqlen + offs_n * stride_A_cumsum_k)

        for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
            dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
            a = tl.load(a_ptrs, mask=(offs_cs < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
            dt = tl.load(dt_ptrs, mask=(offs_m < chunk_size_limit), other=0.0).to(dot_dtype)
            dAm = tl.load(dAm_ptrs, mask=(offs_m < chunk_size_limit), other=0.0).to(dot_dtype)
            dAk = tl.load(dAk_ptrs, mask=(offs_cs < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)

            dA_diff = dAk[None, :] - dAm[:, None]
            exp_term = tl.exp(dA_diff)
            a_dt_prod = a[None, :] * dt[:, None]
            temp = dout * a_dt_prod * exp_term

            mask = offs_m[:, None] <= (offs_cs[None, :] + BLOCK_SIZE_CS * cs)
            temp = tl.where(mask, temp, 0.0)

            acc = tl.multiple_of(acc, 8)
            acc += tl.sum(temp, axis=1)

            dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
            a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen
            dAk_ptrs += BLOCK_SIZE_CS * stride_A_cumsum_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_g * stride_res_group
        res_ptrs = res_ptr + (offs_m * stride_res_seqlen + offs_n * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m < chunk_size_limit)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_g * stride_db_group
    db_ptrs = db_ptr + (offs_m * stride_db_seqlen + offs_n * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m < chunk_size_limit))


# *bmm CABT backward dC*
def _bmm_CABT_dc_chunk_bwd(a, dt, dA_cumsum, dout, residual=None, out=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dout: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
        residual: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    Return:
        out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)

    If there was seq_idx in the fwd pass, then dout[i, j] for seq_idx[i] != seq_idx[j] should already be
    zeroed out before calling this function.
    """
    # Check constraints.
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
        nheads = dA_cumsum.shape[1]
    nchunks, chunk_size = dout.shape[1], dout.shape[-1]
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if dout.stride(-1) != 1 and dout.stride(-2) != 1:
        dout = dout.contiguous()
    if residual is not None:
        assert residual.shape == (batch, seqlen, k) if not has_groups else (batch, seqlen, ngroups, k)
        if residual.stride(-1) != 1 and residual.stride(1) != 1:
            residual = residual.contiguous()
    # Allocates output.
    if out is not None:
        assert out.shape == a.shape
        assert out.stride(-1) == 1 or out.stride(1) == 1
    else:
        out = torch.empty_like(a)
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or dout.dtype == torch.bfloat16 else
                 (tl.float16 if a.dtype == torch.float16 or dout.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * k, batch,
                    nchunks if not has_groups else nchunks * ngroups)
    residual_strides = ((residual.stride(0), residual.stride(1), 0 if not has_groups else residual.stride(2),
                         residual.stride(-1))
                        if residual is not None else (0, 0, 0, 0))
    with torch.cuda.device(a.device.index):
        _bmm_CABT_chunk_dc_bwd_kernel[grid](
            a, dt, dA_cumsum, dout, out, residual,
            seqlen, chunk_size, k, nheads, nheads // ngroups if has_groups else 1, 
            a.stride(0), a.stride(1), 0 if not has_groups else a.stride(2), a.stride(-1),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
            dout.stride(0), dout.stride(1), 0 if not has_groups else dout.stride(2), dout.stride(-2), dout.stride(-1),
            out.stride(0), out.stride(1), 0 if not has_groups else out.stride(2), out.stride(-1),
            residual_strides[0], residual_strides[1], residual_strides[2], residual_strides[3],
            dot_dtype,
            HAS_RESIDUAL=residual is not None,
        )
    return out

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_CS': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_CS': 64}, num_stages=5, num_warps=4),
    ],
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_CABT_chunk_dc_bwd_kernel(
    # Pointers to matrices
    a_ptr, dt_ptr, dA_cumsum_ptr, dout_ptr, db_ptr, res_ptr,
    # Matrix dimensions
    seqlen, chunk_size, K, nheads, nheads_ngroups_ratio,
    stride_a_batch, stride_a_seqlen, stride_a_group, stride_ak,
    stride_dt_batch, stride_dt_head, stride_dt_seqlen, stride_dtk,
    stride_A_cumsum_batch, stride_A_cumsum_head, stride_A_cumsum_seqlen, stride_A_cumsum_k,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_seqlen, stride_db_group, stride_db_k,
    stride_res_batch, stride_res_seqlen, stride_res_group, stride_res_k,
    # Meta-parameters
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_cg = tl.program_id(axis=2)
    pid_c = pid_cg // (nheads // nheads_ngroups_ratio)
    pid_g = pid_cg - pid_c * (nheads // nheads_ngroups_ratio)
    num_pid_n = K
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    acc = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    for i in range(0, nheads_ngroups_ratio):
        pid_h = pid_g * nheads_ngroups_ratio + i

        a_ptr_i = a_ptr + pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_g * stride_a_group
        dA_cumsum_ptr_i = dA_cumsum_ptr + pid_b * stride_A_cumsum_batch + pid_c * chunk_size * stride_A_cumsum_seqlen + pid_h * stride_A_cumsum_head
        dt_ptr_i = dt_ptr + pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen + pid_h * stride_dt_head
        dout_ptr_i = dout_ptr + pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n
        offs_cs = tl.arange(0, BLOCK_SIZE_CS)
        tl.multiple_of(offs_m, 8)
        tl.multiple_of(offs_cs, 8)

        dout_ptrs = dout_ptr_i + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
        a_ptrs = a_ptr_i + (offs_cs * stride_a_seqlen + offs_n * stride_ak)
        dt_ptrs = dt_ptr_i + (offs_cs * stride_dt_seqlen + offs_n * stride_dtk)
        dAm_ptrs = dA_cumsum_ptr_i + (offs_m * stride_A_cumsum_seqlen + offs_n * stride_A_cumsum_k)
        dAk_ptrs = dA_cumsum_ptr_i + (offs_cs * stride_A_cumsum_seqlen + offs_n * stride_A_cumsum_k)

        for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
            dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
            a = tl.load(a_ptrs, mask=(offs_cs < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
            dt = tl.load(dt_ptrs, mask=(offs_cs < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
            dAm = tl.load(dAm_ptrs, mask=(offs_m < chunk_size_limit), other=0.0).to(dot_dtype)
            dAk = tl.load(dAk_ptrs, mask=(offs_cs < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)

            dA_diff = - (dAk[None, :] - dAm[:, None])
            exp_term = tl.exp(dA_diff)
            a_dt_prod = a[None, :] * dt[None, :]
            temp = dout * a_dt_prod * exp_term

            mask = offs_m[:, None] >= (offs_cs[None, :] + BLOCK_SIZE_CS * cs)
            temp = tl.where(mask, temp, 0.0)

            acc = tl.multiple_of(acc, 8)
            acc += tl.sum(temp, axis=1)

            dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
            a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen
            dt_ptrs += BLOCK_SIZE_CS * stride_dt_seqlen
            dAk_ptrs += BLOCK_SIZE_CS * stride_A_cumsum_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_g * stride_res_group
        res_ptrs = res_ptr + (offs_m * stride_res_seqlen + offs_n * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m < chunk_size_limit)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_g * stride_db_group
    db_ptrs = db_ptr + (offs_m * stride_db_seqlen + offs_n * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m < chunk_size_limit))


# *bmm CABT backward dBC*
def _bmm_CABT_dbc_chunk_bwd(B, C, dt, dA_cumsum, dCABT, dB=None, dC=None, dB_out=None, dC_out=None):
    """
    Arguments:
        B: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        C: (batch, seqlen, k) or (batch, seqlen, ngroups, k) 
        dt: (batch, nheads, seqlen, d_state)
        dA_cumsum: (batch, nheads, seqlen, d_state)
        dCABT: (batch, nchunks, chunk_size, nheads, d_state)
        dB: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dC: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dB_out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dC_out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    Returns:
        dB_out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dC_out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    """
    # Check constraints.
    has_groups = B.dim() == 4
    if not has_groups:
        batch, seqlen, k = B.shape
    else:
        batch, seqlen, ngroups, k = B.shape
        nheads = dA_cumsum.shape[1]
    nchunks, chunk_size = dCABT.shape[1], dCABT.shape[-1]
    if B.stride(-1) != 1 and B.stride(1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1 and C.stride(1) != 1:
        C = C.contiguous()
    if dCABT.stride(-1) != 1 and dCABT.stride(-2) != 1:
        dCABT = dCABT.contiguous()
    for _, res in [(dB_out, dB), (dC_out, dC)]:
        if res is not None:
            assert res.shape == (batch, seqlen, k) if not has_groups else (batch, seqlen, ngroups, k)
            if res.stride(-1) != 1 and res.stride(1) != 1:
                res = res.contiguous()
    # Allocates output.
    if dB_out is None:
        dB_out = torch.empty_like(B)
    if dC_out is None:
        dC_out = torch.empty_like(C)

    dot_dtype = (tl.bfloat16 if B.dtype == torch.bfloat16 or dCABT.dtype == torch.bfloat16 else
                 (tl.float16 if B.dtype == torch.float16 or dCABT.dtype == torch.float16 else tl.float32))

    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * k, batch,
                        nchunks if not has_groups else nchunks * ngroups)

    def get_residual_strides(res):
        if res is not None:
            return (res.stride(0), res.stride(1), 
                   0 if not has_groups else res.stride(2),
                   res.stride(-1))
        return (0, 0, 0, 0)

    dB_residual_strides = get_residual_strides(dB)
    dC_residual_strides = get_residual_strides(dC)

    with torch.cuda.device(B.device.index):
        _bmm_CABT_chunk_dbc_bwd_kernel[grid](
            B, C, dt, dA_cumsum, dCABT, dB_out, dC_out, dB, dC,
            seqlen, chunk_size, k, nheads, nheads // ngroups if has_groups else 1,
            B.stride(0), B.stride(1), 0 if not has_groups else B.stride(2), B.stride(-1),
            C.stride(0), C.stride(1), 0 if not has_groups else C.stride(2), C.stride(-1),
            dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
            dCABT.stride(0), dCABT.stride(1), 0 if not has_groups else dCABT.stride(2),
            dCABT.stride(-2), dCABT.stride(-1),
            dB_out.stride(0), dB_out.stride(1), 0 if not has_groups else dB_out.stride(2), dB_out.stride(-1),
            dC_out.stride(0), dC_out.stride(1), 0 if not has_groups else dC_out.stride(2), dC_out.stride(-1),
            *dB_residual_strides, *dC_residual_strides,
            dot_dtype,
            HAS_B_RESIDUAL=dB is not None,
            HAS_C_RESIDUAL=dC is not None,
        )
    return dB_out, dC_out

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_CS': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_CS': 64}, num_stages=5, num_warps=4),
    ],
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_CABT_chunk_dbc_bwd_kernel(
    # Pointers to matrices
    b_ptr, c_ptr, dt_ptr, dA_cumsum_ptr, dout_ptr, 
    db_out_ptr, dc_out_ptr, db_res_ptr, dc_res_ptr,
    # Matrix dimensions
    seqlen, chunk_size, K, nheads, nheads_ngroups_ratio,
    # Strides
    stride_b_batch, stride_b_seqlen, stride_b_group, stride_bk,
    stride_c_batch, stride_c_seqlen, stride_c_group, stride_ck,
    stride_dt_batch, stride_dt_head, stride_dt_seqlen, stride_dtk,
    stride_A_cumsum_batch, stride_A_cumsum_head, stride_A_cumsum_seqlen, stride_A_cumsum_k,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_out_batch, stride_db_out_seqlen, stride_db_out_group, stride_db_out_k,
    stride_dc_out_batch, stride_dc_out_seqlen, stride_dc_out_group, stride_dc_out_k,
    stride_db_res_batch, stride_db_res_seqlen, stride_db_res_group, stride_db_res_k,
    stride_dc_res_batch, stride_dc_res_seqlen, stride_dc_res_group, stride_dc_res_k,
    # Meta-parameters
    dot_dtype: tl.constexpr,
    HAS_B_RESIDUAL: tl.constexpr,
    HAS_C_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_CS: tl.constexpr,
    UNROLL: tl.constexpr = 4,
):
    pid_b = tl.program_id(axis=1)
    pid_cg = tl.program_id(axis=2)
    pid_c = pid_cg // (nheads // nheads_ngroups_ratio)
    pid_g = pid_cg - pid_c * (nheads // nheads_ngroups_ratio)
    num_pid_n = K
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    acc_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_c = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n
    offs_cs = tl.arange(0, BLOCK_SIZE_CS)
    tl.multiple_of(offs_m, 8)
    tl.multiple_of(offs_cs, 8)

    for i in range(0, nheads_ngroups_ratio):
        pid_h = pid_g * nheads_ngroups_ratio + i

        b_base_ptr = b_ptr + pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_g * stride_b_group
        c_base_ptr = c_ptr + pid_b * stride_c_batch + pid_c * chunk_size * stride_c_seqlen + pid_g * stride_c_group
        dA_base_ptr = dA_cumsum_ptr + pid_b * stride_A_cumsum_batch + pid_c * chunk_size * stride_A_cumsum_seqlen + pid_h * stride_A_cumsum_head
        dt_base_ptr = dt_ptr + pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen + pid_h * stride_dt_head
        dout_base_ptr = dout_ptr + pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

        dout_b_ptrs = dout_base_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
        dout_c_ptrs = dout_base_ptr + (offs_m[:, None] * stride_dout_csize_m + offs_cs[None, :] * stride_dout_csize_n)
        b_ptrs = b_base_ptr + (offs_cs * stride_b_seqlen + offs_n * stride_bk)
        c_ptrs = c_base_ptr + (offs_cs * stride_c_seqlen + offs_n * stride_ck)
        dt_b_ptrs = dt_base_ptr + (offs_m * stride_dt_seqlen + offs_n * stride_dtk)
        dt_c_ptrs = dt_base_ptr + (offs_cs * stride_dt_seqlen + offs_n * stride_dtk)
        dAm_ptrs = dA_base_ptr + (offs_m * stride_A_cumsum_seqlen + offs_n * stride_A_cumsum_k)
        dAk_ptrs = dA_base_ptr + (offs_cs * stride_A_cumsum_seqlen + offs_n * stride_A_cumsum_k)

        for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS * UNROLL)):
            for u in range(UNROLL):
                current_cs = cs * UNROLL + u
                if current_cs * BLOCK_SIZE_CS < chunk_size_limit:
                    b_block = tl.load(b_ptrs, mask=(offs_cs < chunk_size_limit - current_cs * BLOCK_SIZE_CS),
                                    other=0.0, eviction_policy="evict_last").to(dot_dtype)
                    c_block = tl.load(c_ptrs, mask=(offs_cs < chunk_size_limit - current_cs * BLOCK_SIZE_CS),
                                    other=0.0, eviction_policy="evict_last").to(dot_dtype)
                    
                    dAm = tl.load(dAm_ptrs, mask=(offs_m < chunk_size_limit),
                                other=0.0, eviction_policy="evict_first").to(dot_dtype)
                    dAk = tl.load(dAk_ptrs, mask=(offs_cs < chunk_size_limit - current_cs * BLOCK_SIZE_CS),
                                other=0.0, eviction_policy="evict_first").to(dot_dtype)
                    
                    dA_diff = dAk[None, :] - dAm[:, None]
                    exp_term = tl.exp(dA_diff)
                    exp_term_neg = 1.0 / exp_term

                    dout_b = tl.load(dout_b_ptrs,
                                   mask=(offs_m[:, None] < chunk_size) & 
                                        (offs_cs[None, :] < chunk_size_limit - current_cs * BLOCK_SIZE_CS),
                                   other=0.0).to(dot_dtype)
                    mask_b = offs_m[:, None] <= (offs_cs[None, :] + BLOCK_SIZE_CS * current_cs)
                    dt_b = tl.load(dt_b_ptrs, mask=(offs_m < chunk_size_limit), other=0.0).to(dot_dtype)
                    
                    temp_b = tl.where(mask_b,
                        tl.fma(dout_b * c_block[None, :], dt_b[:, None] * exp_term, 0.0),
                        0.0)
                    acc_b = tl.multiple_of(acc_b, 8)
                    acc_b += tl.sum(temp_b, axis=1)

                    dout_c = tl.load(dout_c_ptrs,
                                   mask=(offs_m[:, None] < chunk_size) & 
                                        (offs_cs[None, :] < chunk_size_limit - current_cs * BLOCK_SIZE_CS),
                                   other=0.0).to(dot_dtype)
                    mask_c = offs_m[:, None] >= (offs_cs[None, :] + BLOCK_SIZE_CS * current_cs)
                    dt_c = tl.load(dt_c_ptrs,
                                 mask=(offs_cs < chunk_size_limit - current_cs * BLOCK_SIZE_CS),
                                 other=0.0).to(dot_dtype)
                    
                    temp_c = tl.where(mask_c,
                        tl.fma(dout_c * b_block[None, :], dt_c[None, :] * exp_term_neg, 0.0),
                        0.0)
                    acc_c = tl.multiple_of(acc_c, 8)
                    acc_c += tl.sum(temp_c, axis=1)

                    dout_b_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
                    dout_c_ptrs += BLOCK_SIZE_CS * stride_dout_csize_n
                    b_ptrs += BLOCK_SIZE_CS * stride_b_seqlen
                    c_ptrs += BLOCK_SIZE_CS * stride_c_seqlen
                    dt_c_ptrs += BLOCK_SIZE_CS * stride_dt_seqlen
                    dAk_ptrs += BLOCK_SIZE_CS * stride_A_cumsum_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_store = offs_m < chunk_size_limit

    if HAS_B_RESIDUAL:
        db_res_ptr += pid_b * stride_db_res_batch + pid_c * chunk_size * stride_db_res_seqlen + pid_g * stride_db_res_group
        db_res_ptrs = db_res_ptr + (offs_m * stride_db_res_seqlen + offs_n * stride_db_res_k)
        res_b = tl.load(db_res_ptrs, mask=mask_store).to(tl.float32)
        acc_b += res_b
    db_out = acc_b.to(db_out_ptr.dtype.element_ty)
    db_out_ptr += pid_b * stride_db_out_batch + pid_c * chunk_size * stride_db_out_seqlen + pid_g * stride_db_out_group
    db_out_ptrs = db_out_ptr + (offs_m * stride_db_out_seqlen + offs_n * stride_db_out_k)
    tl.store(db_out_ptrs, db_out, mask=mask_store)

    if HAS_C_RESIDUAL:
        dc_res_ptr += pid_b * stride_dc_res_batch + pid_c * chunk_size * stride_dc_res_seqlen + pid_g * stride_dc_res_group
        dc_res_ptrs = dc_res_ptr + (offs_m * stride_dc_res_seqlen + offs_n * stride_dc_res_k)
        res_c = tl.load(dc_res_ptrs, mask=mask_store).to(tl.float32)
        acc_c += res_c
    dc_out = acc_c.to(dc_out_ptr.dtype.element_ty)
    dc_out_ptr += pid_b * stride_dc_out_batch + pid_c * chunk_size * stride_dc_out_seqlen + pid_g * stride_dc_out_group
    dc_out_ptrs = dc_out_ptr + (offs_m * stride_dc_out_seqlen + offs_n * stride_dc_out_k)
    tl.store(dc_out_ptrs, dc_out, mask=mask_store)