import time
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from issm_triton.issm_combined import ISSM_chunk_scan_combined

def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ISSM_minimal_discrete(X, dt, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        dt: (batch, length, n_heads, d_state)
        A: (batch, length, n_heads, d_state)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype == dt.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, dt, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, dt, A, B, C)]

    A = rearrange(A, "b c l h n -> b h n c l")
    A_cumsum = torch.cumsum(A, dim=-1) # .detach()
    A_cumsum.retain_grad()

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    CABT  = torch.einsum("bclhn,bcshn,bhncls,bcshn->bchls", C, B, L, dt)
    Y_diag  = torch.einsum("bchls,bcshp->bclhp", CABT, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, :, -1:] - A_cumsum)) # ddA_next
    states = torch.einsum("bclhn,bhncl,bclhn,bclhp->bchpn", B, decay_states, dt, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    else:
        initial_states = initial_states.unsqueeze(1)
    new_states1 = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, :, -1], (1, 0)))) # ddA_cumsum_prev
    new_states = torch.einsum("bhnzc,bchpn->bzhpn", decay_chunk, new_states1)
    new_states2, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum) # ddA_cumsum_prev
    Y_off = torch.einsum('bclhn,bchpn,bhncl->bclhp', C, new_states2, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


def test_correctness_triton():
    torch.manual_seed(42)

    ## Dimensions
    batch, seqlen, chunk_size, dim, headdim = 1, 512, 256, 256, 32
    nheads = dim // headdim 
    ngroups = 1
    dstate = 256
    dtype = torch.float32
    device = "cuda"

    x_ref = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device).requires_grad_()
    initial_states_ref = torch.randn(batch, nheads, headdim, dstate, dtype=dtype, device=device).requires_grad_()
    dt_ref = F.softplus(torch.randn(batch, seqlen, nheads, dstate, dtype=torch.float32, device=device) - 4).requires_grad_()
    A_ref = (-torch.exp(torch.rand(nheads, dstate, dtype=torch.float32, device=device))).requires_grad_()
    B_ref = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    C_ref = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    y_ref, final_state_ref = ISSM_minimal_discrete(x_ref, dt_ref, A_ref*dt_ref, B_ref, C_ref, chunk_size, initial_states=initial_states_ref)

    x = x_ref.detach().clone().requires_grad_()
    dt = dt_ref.detach().clone().requires_grad_()
    A = A_ref.detach().clone().requires_grad_()
    B = B_ref.detach().clone().requires_grad_()
    C = C_ref.detach().clone().requires_grad_()
    initial_states = initial_states_ref.detach().clone().requires_grad_()
    y, final_state = ISSM_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, initial_states=initial_states, return_final_states=True)
    
    print(f'Output mean diff (%): {((y - y_ref)/ (y_ref)).abs().mean().item()}')
    print(f'Final state mean diff (%): {((final_state - final_state_ref)/ (final_state_ref)).abs().mean().item()}')

    loss_func = torch.nn.MSELoss()
    loss = loss_func(y, torch.ones_like(y)) * 10000
    loss.backward()
    loss_ref = loss_func(y_ref, torch.ones_like(y_ref)) * 10000
    loss_ref.backward()

    print(f'dx mean diff (%): {((x.grad - x_ref.grad) / (x_ref.grad)).abs().mean().item()}')
    print(f'ddt mean diff (%): {((dt.grad - dt_ref.grad) / (dt_ref.grad)).abs().mean().item()}')
    print(f'dA mean diff (%): {((A.grad - A_ref.grad)/ (A_ref.grad)).abs().mean().item()}')
    print(f'dB mean diff (%): {((B.grad - B_ref.grad) / (B_ref.grad)).abs().mean().item()}')
    print(f'dC mean diff (%): {((C.grad - C_ref.grad) / (C_ref.grad)).abs().mean().item()}')
    print(f'dinitial_states mean diff (%): {((initial_states.grad - initial_states_ref.grad) / (initial_states_ref.grad)).abs().mean().item()}')

def test_speed_triton():
    torch.manual_seed(42)

    ## Dimensions
    batch, seqlen, chunk_size, dim, headdim = 1, 512, 256, 256, 32
    nheads = dim // headdim 
    ngroups = 1
    dstate = 256
    dtype = torch.float32
    device = "cuda"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device).requires_grad_()
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dstate, dtype=torch.float32, device=device) - 4).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dstate, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device).requires_grad_()
    initial_states = torch.randn(batch, nheads, headdim, dstate, dtype=dtype, device=device).requires_grad_()

    for i in range(100):
        start = time.process_time()
        y, final_state = ISSM_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, initial_states=initial_states, return_final_states=True)
        end = time.process_time()
        print(f'{i}: Time taken: {end - start} seconds')


if __name__ == "__main__":
    test_correctness_triton()
    test_speed_triton()