using QuantumOptics
using Arpack






function psi_to_proj(psi)
    """
    Generates an operator that projects onto the given state

    INPUTS:
        - psi (QuantumOptics Ket): state vector to project onto

    RETURNS:
        - QuantumOptics operator that projects onto psi
    """
    return tensor(psi, dagger(psi))
end

# Cosine in terms of complex exponentials
function cos_operator(x)
    """
    Returns the cosine of an operator, generated using 
    matrix exponentiation

    INPUTS:
        - x (QuantumOptics Operator): operator to be cos-ed

    RETURNS:
        QuantumOptics operator cos(x) = (exp(i*x)+exp(-i*x))/2
    """
    return (exp(1.0im*dense(x)) + exp(-1.0im*dense(x)))/2
end

function sin_operator(x)
    """
    Returns the sine of an operator, generated using 
    matrix exponentiation

    INPUTS:
        - x (QuantumOptics Operator): operator to be cos-ed

    RETURNS:
        QuantumOptics operator sin(x) = (exp(i*x)-exp(-i*x))/2i
    """
    return (exp(1.0im*dense(x)) - exp(-1.0im*dense(x)))/(2.0im)
end

function H_transmon_single(E_j, E_c, n_g, phi_range=(-2*pi, 2*pi), n_points=31)
    """
    Generates a Hamiltonian for a single transmon
    
    INPUTS: 
        - E_j (Float64): Josephson energy of junction (GHz/h)
        - E_c (Float64): Charging energy of capacitor (GHz/h)
        - n_g (Float64): offset charge (dimensionless)
        - phi_range (List of Float64): min and max phi for computational space
        - n_points (Integer): number of points in the discrete phi space
    
    RETURNS:
        - LazySum QuantumOptics Operator for the Hamiltonian. Wrap call in 
            dense() or sparse() to get the non-lazy operator

    """

    # Make the phi/n operators
    phi_hat = transmon_phi_op(phi_range, n_points)
    n_hat = transmon_n_op(phi_range, n_points)

    # Make the Hamiltonians -- Do a Lazy Add: need to convert to dense before
    # doing most things
    Id = identityoperator(n_hat)

    
    V = cos_operator(phi_hat) 
    K = n_hat^2

    # Third/fourth term is for a drive
    return LazySum([-E_j,  4*E_c, -8*E_c*n_g, 4*E_c*n_g^2], hermitify.([V, K, n_hat, Id]))

end


function set_transmon_params!(H, E_j, E_c, n_g)
    """
    Modifies parameters for a single transmon Hamiltonian
    
    INPUTS: 
        - H (QuantumOptics LazySum Operator): Hamiltonian for the system
        - E_j (Float64): Josephson energy of junction (GHz/h)
        - E_c (Float64): Charging energy of capacitor (GHz/h)
        - n_g (Float64): offset charge (dimensionless)

    """

    H.factors = [-E_j, 4*E_c,  -8*n_g*E_c, 4*E_c*n_g^2]

    return
end

function transmon_phi_op(phi_range = (-2*pi, 2*pi), n_points=31)
    """
    Generates a phi operator for a transmon system
    
    INPUTS: 
        - phi_range (List of Float64): min and max phi for computational space
        - n_points (Integer): number of points in the discrete phi space
    
    RETURNS:
        - QuantumOptics Operator for phi 

    """
    b_phi = PositionBasis(phi_range[1], phi_range[2], n_points)
    return position(b_phi)
end

function transmon_n_op(phi_range = (-2*pi, 2*pi), n_points=31)
    """
    Generates an n operator for a transmon system
    
    INPUTS: 
        - phi_range (List of Float64): min and max phi for computational space
        - n_points (Integer): number of points in the discrete phi space
    
    RETURNS:
        - QuantumOptics Operator for n 

    """
    b_phi = PositionBasis(phi_range[1], phi_range[2], n_points)
    return momentum(b_phi)
end

function H_fluxonium_single(E_j, E_l, E_c, phi_e, phi_range = (-3*pi, pi), n_points=31)
    """
    Generates a Hamiltonian for a single fluxonium
    
    INPUTS: 
        - E_j (Float64): Josephson energy of junction (GHz/h)
        - E_l (Float64): Inductive energy of inductor (GHz/h)
        - E_c (Float64): Charging energy of capacitor (GHz/h)
        - phi_e (Float64): external flux (units of flux quantum)
        - phi_range (List of Float64): min and max phi for computational space
        - n_points (Integer): number of points in the discrete phi space
    
    RETURNS:
        - LazySum QuantumOptics Operator for the Hamiltonian. Wrap call in 
            dense() or sparse() to get the non-lazy operator

    """

    # Make the phi/n operators
    phi_hat = fluxonium_phi_op(phi_range, n_points)
    n_hat = fluxonium_n_op(phi_range, n_points)

    # Make the Hamiltonians -- Do a Lazy Add: need to convert to dense before
    # doing most things
    Id = identityoperator(phi_hat)
    H = LazySum([0, 4*E_c,  E_l*2*pi*phi_e, (1/2)*E_l, -E_j, (1/2)*(2*pi*phi_e)^2],
                hermitify.([n_hat, n_hat^2, phi_hat, phi_hat^2, cos_operator(phi_hat), Id]))
    
    # V = -E_j*cos_operator(phi_hat) + (1/2)*E_l*(phi_hat + Id*2*pi*phi_e)^2
    # K = 4*E_c*n_hat^2
    # H =  K + V

    return H

end

function fluxonium_phi_op(phi_range = (-3*pi, pi), n_points=31)
    """
    Generates an n operator for a fluxonium system
    
    INPUTS: 
        - phi_range (List of Float64): min and max phi for computational space
        - n_points (Integer): number of points in the discrete phi space
    
    RETURNS:
        - QuantumOptics Operator for phi

    """
    # Make the basis and phi/n operators
    b_phi = PositionBasis(phi_range[1], phi_range[2], n_points)
    return position(b_phi)
end

function fluxonium_n_op(phi_range = (-3*pi, pi), n_points=31)
    """
    Generates an n operator for a fluxonium system
    
    INPUTS: 
        - phi_range (List of Float64): min and max phi for computational space
        - n_points (Integer): number of points in the discrete phi space
    
    RETURNS:
        - QuantumOptics Operator for fluxonium

    """
    # Make the basis and phi/n operators
    b_phi = PositionBasis(phi_range[1], phi_range[2], n_points)
    return momentum(b_phi)
end

function set_fluxonium_params!(H, E_j, E_l, E_c, phi_e)
    """
    Modifies parameters for a single fluxonium Hamiltonian
    
    INPUTS: 
        - H (QuantumOptics LazySum Operator): Hamiltonian for the system
        - E_j (Float64): Josephson energy of junction (GHz/h)
        - E_l (Float64): Inductive energy of inductor (GHz/h)
        - E_c (Float64): Charging energy of capacitor (GHz/h)
        - n_g (Float64): offset charge (dimensionless)

    """

    H.factors = [0, 4*E_c,  E_l*2*pi*phi_e, (1/2)*E_l, -E_j, (1/2)*(2*pi*phi_e)^2]

    return
end


function hermitify(H)
    """
    Returns a hermitified version of an operator (O+dagger(O))/2. 
    For making operators explicitly Hermitian after numerical issues.

    INPUTS:
        - H (QuantumOptics Operator): operator to hermitify

    RETURNS:
        - (H + dagger(H))/2

    """
    # Make it Hermitian after numerical issues
    return (H+dagger(H))/2
end

function mat_elem(psi1, O, psi2)
    """
    Returns the matrix element <psi1|O|psi2>

    INPUTS:
        - psi1 (QuantumOptics Ket): left state
        - O (QuantumOptics Operator): Operator
        - psi2 (QuantumOptics Ket): right state
    """
    return dagger(psi1)*O*psi2
end


function mat_elems(states, op)
    """
    Generates a matrix that represents the operator op, projected
    into the space made by the given states. Each entry is:

    []_ij = <states[i]|op|states[j]>

    INPUTS:
        - states (list of QuantumOptics Ket): list of state vectors
        - op (QuantumOptics Operator): operator to evaluate

    RETURNS:
        ComplexF64 matrix 

    """
    to_return = zeros(ComplexF64, size(states)[1], size(states)[1])
    for (i,si) in enumerate(states)
        for (j,sj) in enumerate(states)
            to_return[i,j] = mat_elem(si, op, sj)
        end
    end
    return to_return
end	

function trunc_subspace_coupled(esA, nA, esB, nB, es)
    """
    Returns an operator that truncates the coupled space of two systems 
    up to a specified number of eigenstates in each system 

    INPUTS:
        - esA (list of QuantumOptics Kets): eigenstates of the single qubit A 
        - nA (Integer): how many eigenstates to include for qubit A
        - esB (list of QuantumOptics Kets): eigenstates of the single qubit B
        - nB (Integer): how many eigenstates to include for qubit B
        - es (list of QuantumOptics Kets): eigenstates of the composite (coupled) system

    RETURNS:
        - QuantumOptics operator that does the projection. Multiply a ket on 
            the right to project it into the trucated spate.
        - QuantumOptics Basis object for the new space

    """
		
    all_states = [[tensor(esA[i], esB[j]) for j in range(1,nB)] for i in range(1,nA)]
    comp_basis = SubspaceBasis(collect(Iterators.flatten(all_states)))

    return projector(comp_basis, es[1].basis), comp_basis
end

function trunc_subspace(states)
    """
    Returns an operator that truncates the space of a single
    qubit into the states given in the list states

    INPUTS:
        - states (list of QuantumOptics Kets): states to truncate into a basis of

    RETURNS:
        - QuantumOptics operator that does the projection. Multiply a ket on 
            the right to project it into the trucated spate.
        - QuantumOptics Basis object for the new space

    """
    comp_basis = SubspaceBasis(states)
    return projector(comp_basis, basis(states[1])), comp_basis
end


function trunc_op(proj, op)
    """
    Project an operator into the truncated subspace defined by proj

    INPUTS:
        - proj (QuantumOptics Operator): projector created by trunc_subspace
        - op (QuantumOptics Operator): operator to truncate

    RETURNS:
        - op projected into the subspace defined by proj
    """
    return proj*op*dagger(proj)
end


function comp_state(i,j, comp_basis, n_comp_i, n_comp_j)
    """
    Creates a computational state |i>|j> in a truncated coupld basis
    of two qubits

    INPUTS: 
        - i (Integer): state of first qubit 
        - j (Integer): state of second qubit
        - comp_basis (QuantumOptics Basis): basis for the truncated space
        - n_comp_i (Integer): How many eigenstates were included for the first qubit
        - n_comp_j (Integer): How many eigenstates were included for the second qubit

    RETURNS:
        - product state |i>|j> in the truncated space
    """
    return Ket(comp_basis, one_vec(n_comp_i*n_comp_j,n_comp_j*i+j+1))
end

function one_vec(N,n)
    """
    Creates a vector of length N with all zeros except for a single
    one in the n'th spot.
    
    INPUTS:
        - N (Integer): length of the vector
        - n (Integer): index to make 1
    
    RETURNS:
        - ComplexF64 vector will all zeros, except for a 1 in the 
            n'th index

    """
    v = zeros(ComplexF64, N)
    v[n] = 1
    return v
end




function H_zero_pi(E_j, dE_j, Ecθ, Ecϕ, E_l, ngθ, phi_ext,
     θ_basis = PositionBasis(-pi/2, 3*pi/2, 50), ϕ_basis = PositionBasis(-4*pi, 4*pi, 50))
    """
    Generates a Hamiltonian for the "soft Zero-Pi" 
    see: https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.010339

    INPUTS:
        - E_j (Float64): Josephson energy (GHz/h)
        - dE_j (Float64): Relative junction-energy asymetry -- Not currently included -- (GHz/h)
        - Ecθ (Float64): Charging energy for theta mode (GHz/h)
        - Ecϕ (Float64): Charging energy for the phi mode (Ghz/h)
        - E_l (Float64): Inductive energy (GHz/h)
        - ngθ (Float64): Offset charge for theta mode (dimensionless)
        - phi_ext (Float64): External flux (units of flux quantum) 
        - θ_basis (QuantumOptics Basis): Flux basis for the theta mode 
        - ϕ_basis (QuantumOptics Basis): Flux basis for the phi mode

    RETURNS:
        - Zero pi Hamiltonian as a sparse QuantumOptics operator

    """

    # Define Operators
    ϕ_op = position(ϕ_basis)
    nϕ = momentum(ϕ_basis)
    id_op_ϕ = identityoperator(ϕ_op)
    
    θ_op = position(θ_basis)
    nθ = momentum(θ_basis)
    id_op_θ = identityoperator(θ_op)

    # Uncoupled and coupled parts of the hamiltonian
    # Hθ =  LazySum([1.0, ngθ, ngθ^2],
    #    hermitify.([nθ^2, -2*nθ, id_op_θ]))
    
    # Hϕ = LazySum([4*Ecϕ, E_l, 0.0, 0.0, 0.0],
    #   hermitify.([nϕ^2, ϕ_op^2, nϕ, ϕ_op, id_op_ϕ]))

    # Coupled part of the Hamiltonian
    # cosθ = cos_operator(θ_op)
    # sinθ = sin_operator(θ_op)
    # cosϕ =  cos_operator(ϕ_op)
    # sinϕ = sin_operator(ϕ_op)
    # Hc = LazySum([cos(phi_ext), sin(phi_ext), 0*cos(phi_ext), 0*sin(phi_ext)],
    #  hermitify.([-2*tensor(cosθ,cosϕ), -2*tensor(cosθ,sinϕ), dE_j*tensor(sinθ, sinϕ),-dE_j*tensor(sinθ, cosϕ)]))

    Hθ = 4*Ecθ*(nθ-ngθ*id_op_θ)^2
    Hϕ_kin = 4*Ecϕ*nϕ^2
    Hϕ_pot = E_l*ϕ_op^2 
    Hc = -2*E_j*tensor(cos_operator(θ_op), cos_operator(ϕ_op-id_op_ϕ*phi_ext))

    # Make the full Hamiltonian
    H = LazySum([1.0, 1.0, 1.0, 1.0],
                hermitify.([tensor(dense(Hθ), id_op_ϕ),tensor(id_op_θ,dense(Hϕ_kin)),tensor(id_op_θ,dense(Hϕ_pot)),dense(Hc)]))

    return H
end




function two_qubit_system(Ha, Hb, Hc, Hda, Hdb)
    """
    Creates a two qubit system out of single qubit Hamiltonians. 
    Returns it as a LazySum, with the drive terms as the first (qubit A)
    and second (qubit B) operators in the sum. The third term contains
    the individual qubits + coupling. 

    INPUTS:
        - Ha (QuantumOptics Operator): Hamiltonian for qubit A
        - Hb (QuantumOptics Operator): Hamiltonian for qubit B
        - Hc (QuantumOptics Operator): Coupling Hamiltonian
        - Hda (QuantumOptics Operator): Drive on qubit A
        - Hdb (QuantumOptics Operator): Drive on qubit B
    
    RETURNS:
        - QuantumOptics LazySum Hamiltonian for the composite system
    """

    id_a = sparse(identityoperator(Ha))
    id_b = sparse(identityoperator(Hb))
    return LazySum([0.0, 0.0, 1.0], hermitify.([Hda, Hdb, tensor(id_b, sparse(Hb))+tensor(sparse(Ha), id_a)+Hc]))

end


function find_prod_state(psi, esA, esB)
    """
    Find the product state that corresponds most closely to psi. Returns -1
    if none is found.

    INPUTS:
        - psi (QuantumOptics Ket): the state you want to identify
        - esA (QuantumOptics Ket): uncoupled eigenstates of qubit A
        - esB (QuantumOptics Ket): uncoupled eigenstates of qubit B
    
    RETURNS:
        - (i,j) indices of states that this corresponds to

    """
    psi_dag = dagger(psi)
    for (i, sA) in enumerate(esA)
        for (j, sB) in enumerate(esB)
            overlap = abs(psi_dag*tensor(sA, sB))
            if overlap >= 0.5
                return (i,j), overlap
            end
        end
    end

    return -1

end


function get_transition_freq(H, s1, s2)
    return abs(expect(H, s1) - expect(H, s2))
end
    












