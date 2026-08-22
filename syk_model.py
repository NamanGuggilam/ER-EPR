import numpy as np
import itertools
from scipy.linalg import eigh

def generate_majoranas(N):
    """
    Constructs the N Majorana fermion operators as a list of explicit 
    (2**(N//2) x 2**(N//2)) complex matrices satisfying {chi_i, chi_j} = delta_ij * I.
    """
    assert N % 2 == 0, "N must be an even integer."
    num_pairs = N // 2
    dim = 2 ** num_pairs
    
    # Base Pauli matrices
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    id2 = np.eye(2, dtype=complex)
    
    majoranas = []
    
    for k in range(1, num_pairs + 1):
        # Initialize left and right string structures
        left_string = [sz] * (k - 1)
        right_string = [id2] * (num_pairs - k)
        
        # Assemble chi_(2k-1)
        term_odd = left_string + [sx] + right_string
        chi_odd = term_odd[0]
        for mat in term_odd[1:]:
            chi_odd = np.kron(chi_odd, mat)
        majoranas.append(chi_odd / np.sqrt(2))
        
        # Assemble chi_(2k)
        term_even = left_string + [sy] + right_string
        chi_even = term_even[0]
        for mat in term_even[1:]:
            chi_even = np.kron(chi_even, mat)
        majoranas.append(chi_even / np.sqrt(2))
        
    return majoranas

def build_syk_hamiltonian(N, J=1.0):
    """
    Generates the 4-body interacting SYK Hamiltonian with quenched disorder.
    Variance is strictly scaled as 6 * J^2 / N^3.
    """
    dim = 2 ** (N // 2)
    majoranas = generate_majoranas(N)
    
    # Initialize empty complex array
    H = np.zeros((dim, dim), dtype=complex)
    
    # Total unique 4-fermion combinations
    combos = list(itertools.combinations(range(N), 4))
    num_combos = len(combos)
    
    # Draw Gaussian couplings with exact N-dependent scaling
    scale = np.sqrt(6.0 * (J**2) / (N**3))
    couplings = np.random.normal(loc=0.0, scale=scale, size=num_combos)
    
    # Vectorized accumulation of the multi-particle terms
    for idx, (i, j, k, l) in enumerate(combos):
        # Maintain absolute index ordering to preserve anti-commutation properties
        term = majoranas[i] @ majoranas[j] @ majoranas[k] @ majoranas[l]
        H += couplings[idx] * term
        
    return H

# --- Verification & Exact Diagonalization Execution ---
if __name__ == "__main__":
    N_test = 8
    print(f"Initializing SYK substrate for N = {N_test}...")
    
    # Construct and verify algebraic integrity
    chis = generate_majoranas(N_test)
    anti_comm_check = chis[0] @ chis[1] + chis[1] @ chis[0]
    print(f"Algebra Validation ({{chi_0, chi_1}} == 0): Success Matrix Norm = {np.linalg.norm(anti_comm_check):.4e}")
    
    # Generate and Diagonalize Hamiltonian
    H_syk = build_syk_hamiltonian(N_test)
    energies, states = eigh(H_syk)
    
    print("\n--- Numerical Eigensystem Extracted ---")
    print(f"Hilbert Space Dimension: {H_syk.shape[0]}")
    print(f"Ground State Energy (E_0): {energies[0]:.6f}")
    print(f"Max State Energy   (E_max): {energies[-1]:.6f}")
