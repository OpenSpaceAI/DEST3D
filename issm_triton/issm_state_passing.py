import torch
import triton
import triton.language as tl

def _state_passing_fwd(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None,
                       out_dtype=None):
    batch, nchunks, nheads, h_dim, d_state = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks, d_state)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, h_dim, d_state)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, h_dim, d_state), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, h_dim, d_state), device=states.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(d_state, META['BLOCK_SIZE']), batch, nheads * h_dim)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states, out, final_states, dA_chunk_cumsum, initial_states, seq_idx,
            d_state, h_dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            final_states.stride(0), final_states.stride(1), final_states.stride(2), final_states.stride(3),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1), dA_chunk_cumsum.stride(3),
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2), initial_states.stride(3))
              if initial_states is not None else (0, 0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out, final_states


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['d_state'],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr, initstates_ptr, seq_idx_ptr,
    # Matrix dimensions
    d_state, h_dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_hdim, stride_out_dstate,
    stride_final_states_batch, stride_final_states_head, stride_final_states_hdim, stride_final_states_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_dstate,
    stride_initstates_batch, stride_initstates_head, stride_initstates_hdim, stride_initstates_dstate,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2) // h_dim
    pid_hd = tl.program_id(axis=2) % h_dim
    pid_d = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head + pid_hd * stride_states_hdim
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + pid_hd * stride_out_hdim
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head + pid_hd * stride_final_states_hdim
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head + pid_hd * stride_initstates_hdim
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_d * stride_states_dstate
    dA_cs_ptrs = dA_cs_ptr + offs_d * stride_dA_cs_dstate
    out_ptrs = out_ptr + offs_d * stride_out_dstate
    final_states_ptrs = final_states_ptr + offs_d * stride_final_states_dstate

    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_d * stride_initstates_dstate
        states = tl.load(initstates_ptrs, mask=offs_d < d_state, other=0.0).to(tl.float32)
    tl.store(out_ptrs, states, mask=offs_d < d_state)
    out_ptrs += stride_out_chunk
    seq_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_d < d_state, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptrs, mask=offs_d < d_state, other=0.0).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_d < d_state)
        else:
            tl.store(final_states_ptrs, states, mask=offs_d < d_state)
        states_ptrs += stride_states_chunk
        dA_cs_ptrs += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_bwd(
        states, dA_chunk_cumsum, dout, dfinal_states=None, seq_idx=None, has_initial_states=None,
        dstates_dtype=None, states_dtype=None, chunk_size=None
):
    """
    states contains the initial_states at index 0. The final states are not included in states.
    """
    batch, nchunks, nheads, h_dim, d_state = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks, d_state)
    assert dout.shape == (batch, nchunks, nheads, h_dim, d_state)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    dstates = torch.empty_like(dout, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(states, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None
    if has_initial_states:
        dinitstates = torch.empty_like(dstates[:, 0])
    else:
        dinitstates = None
    if dfinal_states is not None:
        assert dfinal_states.shape == (batch, nheads, h_dim, d_state)
    ddA_chunk_cumsum = torch.zeros(batch, nheads, nchunks, h_dim, d_state, dtype=torch.float32, device=dA_chunk_cumsum.device)
    grid = lambda META: (triton.cdiv(d_state, META['BLOCK_SIZE']), batch, nheads * h_dim)
    with torch.cuda.device(dout.device.index):
        _state_passing_bwd_kernel[grid](
            dout, states, dA_chunk_cumsum, dfinal_states, seq_idx,
            dstates, ddA_chunk_cumsum, dinitstates, states_converted,
            d_state, h_dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1), dA_chunk_cumsum.stride(3),
            *((dfinal_states.stride(0), dfinal_states.stride(1), dfinal_states.stride(2), dfinal_states.stride(3))
                if dfinal_states is not None else (0, 0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            ddA_chunk_cumsum.stride(0), ddA_chunk_cumsum.stride(2), ddA_chunk_cumsum.stride(1), ddA_chunk_cumsum.stride(3), ddA_chunk_cumsum.stride(4),
            *((dinitstates.stride(0), dinitstates.stride(1), dinitstates.stride(2), dinitstates.stride(3))
              if dinitstates is not None else (0, 0, 0, 0)),
            CONVERT_STATES=states_converted is not None,
            HAS_DFINAL_STATES=dfinal_states is not None,
            HAS_DINITSTATES=dinitstates is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states
    return (dstates, ddA_chunk_cumsum, dinitstates) if states_dtype is None else (dstates, ddA_chunk_cumsum, dinitstates, states_converted)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['d_state'],
)
@triton.jit
def _state_passing_bwd_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, dA_cs_ptr, dfinal_states_ptr, seq_idx_ptr,
    dstates_ptr, ddA_cs_ptr, dinitstates_ptr, states_converted_ptr,
    # Matrix dimensions
    d_state, h_dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_hdim, stride_dout_dstate,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_hdim, stride_out_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_dstate,
    stride_dfinal_states_batch, stride_dfinal_states_head, stride_dfinal_states_hdim, stride_dfinal_states_dstate,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_hdim, stride_ddA_cs_dstate,
    stride_dinitstates_batch, stride_dinitstates_head, stride_dinitstates_hdim, stride_dinitstates_dstate,
    # Meta-parameters
    CONVERT_STATES: tl.constexpr,
    HAS_DFINAL_STATES: tl.constexpr,
    HAS_DINITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2) // h_dim
    pid_hd = tl.program_id(axis=2) % h_dim
    pid_d = tl.program_id(axis=0)

    offs_d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_d < d_state
    
    dstates_ptr += pid_b * stride_dstates_batch + pid_h * stride_dstates_head + (nchunks - 1) * stride_dstates_chunk + pid_hd * stride_dstates_hdim
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head + (nchunks - 1) * stride_dA_cs_chunk
    ddA_cs_ptr += pid_b * stride_ddA_cs_batch + pid_h * stride_ddA_cs_head + (nchunks - 1) * stride_ddA_cs_chunk + pid_hd * stride_ddA_cs_hdim
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + (nchunks - 1) * stride_out_chunk + pid_hd * stride_out_hdim
    dout_ptr += pid_b * stride_dout_batch + pid_h * stride_dout_head + (nchunks - 1) * stride_dout_chunk + pid_hd * stride_dout_hdim

    if CONVERT_STATES:
        states_converted_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + (nchunks - 1) * stride_out_chunk + pid_hd * stride_out_hdim
    if HAS_DFINAL_STATES:
        dfinal_states_ptr += pid_b * stride_dfinal_states_batch + pid_h * stride_dfinal_states_head + pid_hd * stride_dfinal_states_hdim
    if HAS_DINITSTATES:
        dinitstates_ptr += pid_b * stride_dinitstates_batch + pid_h * stride_dinitstates_head + pid_hd * stride_dinitstates_hdim
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    dstates_ptrs = dstates_ptr + offs_d * stride_dstates_dstate
    out_ptrs = out_ptr + offs_d * stride_out_dstate
    dout_ptrs = dout_ptr + offs_d * stride_dout_dstate
    dA_cs_ptrs = dA_cs_ptr + offs_d * stride_dA_cs_dstate
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_d * stride_out_dstate

    if HAS_DFINAL_STATES:
        dstates = tl.load(dfinal_states_ptr + offs_d * stride_dfinal_states_dstate, mask=mask, other=0.0).to(tl.float32)
    else:
        dstates = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    tl.store(dstates_ptrs, dstates, mask=mask)
    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + (seqlen - 1) * stride_seq_idx_seqlen)
    dstates_ptrs -= stride_dstates_chunk
    for c in range(nchunks - 1):
        dA_cs = tl.load(dA_cs_ptrs, mask=mask, other=0.0).to(tl.float32)
        out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)

        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (((nchunks - c - 1) * chunk_size - 1) * stride_seq_idx_seqlen))
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new

        ddA = tl.fma(out, dstates, 0.0) * scale
        dstates = tl.fma(scale, dstates, dout)
        tl.store(ddA_cs_ptr + offs_d * stride_ddA_cs_dstate, ddA, mask=mask)
        tl.store(dstates_ptrs, dstates, mask=mask)

        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=mask)

        dout_ptrs -= stride_dout_chunk
        dstates_ptrs -= stride_dstates_chunk
        dA_cs_ptrs -= stride_dA_cs_chunk
        ddA_cs_ptr -= stride_ddA_cs_chunk
        out_ptrs -= stride_out_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk

    if CONVERT_STATES:
        out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(states_converted_ptrs, out, mask=mask)
    if not HAS_DINITSTATES:
        tl.store(ddA_cs_ptr + offs_d * stride_ddA_cs_dstate, 0.0, mask=mask)
    else:
        dA_cs = tl.load(dA_cs_ptrs, mask=mask).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            scale = tl.where(seq_idx == 0, scale, 0.0)

        out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
        ddA = tl.fma(out, dstates, 0.0) * scale
        dout = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)
        dstates = tl.fma(scale, dstates, dout)

        tl.store(ddA_cs_ptr + offs_d * stride_ddA_cs_dstate, ddA, mask=mask)
        tl.store(dinitstates_ptr + offs_d * stride_dinitstates_dstate, dstates, mask=mask)
