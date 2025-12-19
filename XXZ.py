import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh

# Pauli matrices
sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float64))
sy = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.float64))
I2 = identity(2, format="csr", dtype=np.complex128)

# Spin-1/2 operators S = sigma/2
Sx = (sx.astype(np.complex128)) * 0.5
Sy = sy * 0.5
Sz = (sz.astype(np.complex128)) * 0.5

def op_on_site(op, site, N):
    out = None
    for i in range(N):
        piece = op if i == site else I2
        out = piece if out is None else kron(out, piece, format="csr")
    return out

def two_site(op1, i, op2, j, N):
    out = None
    for k in range(N):
        if k == i:
            piece = op1
        elif k == j:
            piece = op2
        else:
            piece = I2
        out = piece if out is None else kron(out, piece, format="csr")
    return out

def xxz_full_hamiltonian(N, J=1.0, Delta=1.0, pbc=False):
    dim = 2**N
    H = csr_matrix((dim, dim), dtype=np.complex128)

    bonds = [(i, i+1) for i in range(N-1)]
    if pbc and N > 2:
        bonds.append((N-1, 0))

    for (i, j) in bonds:
        H += J * (two_site(Sx, i, Sx, j, N) +
                  two_site(Sy, i, Sy, j, N) +
                  Delta * two_site(Sz, i, Sz, j, N))
    return H

def ground_state_full(N, J=1.0, Delta=1.0, pbc=False):
    H = xxz_full_hamiltonian(N, J=J, Delta=Delta, pbc=pbc)
    # smallest algebraic eigenvalue
    vals, vecs = eigsh(H, k=1, which="SA")
    E0 = float(np.real(vals[0]))
    psi0 = vecs[:, 0]  # normalized
    return E0, psi0, H

if __name__ == "__main__":
    N = 10
    J = 1.0
    Delta = 1.0  # Heisenberg
    E0, psi0, H = ground_state_full(N, J=J, Delta=Delta, pbc=False)
    print("E0 =", E0)
    np.save(f"./heisenberg/heisenberg_N{N}_Jz{Delta}.npy", psi0)
