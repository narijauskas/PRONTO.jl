using PyPlot
using QuantumOptics
using LinearAlgebra
using LaTeXStrings
using PlutoUI
using LazyGrids: ndgrid_array
import PlutoUI: combine
include("cqed.jl")


begin
	# Parameters from https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.010339
	# Units of GHz*h
	c = 2*pi
	Ec_theta = 0.092*c
	Ec_phi = 1.142*c
	Ej = 6.013*c
	dEj = 0.1*c
	El = 0.377*c
	beta_phi = 0.27*c
	beta_theta = c*6.6*10^-3
	ng_theta = 0.25
	phi_ext = 0.0
	phi_range = [-4*pi, 4*pi]
	theta_range = [-pi/2, 3*pi/2]
	n_points = 50

	# Define bases
	phi_basis = PositionBasis(phi_range[1], phi_range[2], n_points)
	phi_points = samplepoints(phi_basis)
	theta_basis = PositionBasis(theta_range[1], theta_range[2], n_points)
	theta_points = samplepoints(theta_basis)

	

	# for plotting
	Y,X = ndgrid_array(theta_points,phi_points)

	# Make the ZP Hamiltonian
	Hzp = H_zero_pi(Ej, dEj, Ec_theta, Ec_phi, El, ng_theta, phi_ext, theta_basis, phi_basis)
	V = reshape(diag(real.((Hzp.operators[3]+Hzp.operators[4]).data)),n_points,n_points)
	Hzp = sparse(Hzp)
	
	# Define some operators
	n_theta = momentum(theta_basis)
	id_theta = identityoperator(n_theta)
	n_phi = momentum(phi_basis)
	id_phi = identityoperator(n_phi)
	
	# Make the drive Hamiltonian
	Hdrive = sparse(beta_phi*tensor(id_theta,n_phi) + beta_theta*tensor(n_theta, id_phi))
end



begin
	ev, es = eigenstates(Hzp, 50)
	shift = ev[1]
	ev = real.(ev .- shift)
end



begin
	plt.close("all")
	surf = plot_surface(X,Y,V, cmap="plasma")
	plt.xlim((-3*pi,3*pi))
	plt.xlabel(L"\phi",fontsize=20)
	plt.ylabel(L"\theta",fontsize=20)
	plt.gca().set_zlabel(L"V(\theta, \phi)",fontsize=16)
	plt.colorbar(surf,shrink=0.4)
	# plt.savefig("zp_potential.svg")
	plt.gcf().set_size_inches(10,10)
	plt.gca().tick_params(axis="both", labelsize=14)
	# plt.tight_layout()
	plt.gcf()
end



begin
	function plot_mat_elems(mat, to_show, title = L"$\log_{10}H_{drive}$")
		plt.close("all")
		to_plot = mat_elems(es[to_show], Hdrive)
		plt.imshow(log10.(abs.(to_plot)))
		plt.title(title)
		plt.colorbar()
		kets = [L"$|"*string(x-1)*L"\rangle$" for x in to_show]
		plt.gca().set_xticks(range(0,length(to_show)-1))
		plt.gca().set_xticklabels(kets)
		plt.gca().set_yticks(range(0,length(to_show)-1))
		plt.gca().set_yticklabels(kets)
		plt.gcf()
	end
	plot_mat_elems(Hdrive, [1,3,10])
end



begin
	all_elems = abs.(mat_elems(es, Hdrive))
	states_to_plot = [1,3,10]
	imax = zeros(size(states_to_plot)[1])
	plt.close("all")
	for (i, state) in enumerate(states_to_plot)
		y = abs.(all_elems[state,:])
		plt.plot(y, label = L"$|"*string(state-1)*L"\rangle$",alpha=0.8)
		imax[i] = findmax(y)[2]
	end
	plt.xlabel(L"State Number $n$")
	plt.ylabel(L"$|\langle n |H_{drive}| i \rangle|$ for specified state $i$")
	plt.legend()
	plt.gcf()
end



begin
	plt.close("all")
	to_show = [1,3,10,12,19]
	# to_show = [1,3,10]
	to_plot = mat_elems(es[to_show], Hdrive)
	plt.imshow(log10.(abs.(to_plot)))
	# plt.imshow(abs.(to_plot))
	plt.title(L"$\log_{10}H_{drive}$")
	plt.colorbar()
	kets = [L"$|"*string(x-1)*L"\rangle$" for x in to_show]
	plt.gca().set_xticks(range(0,length(to_show)-1))
	plt.gca().set_xticklabels(kets)
	plt.gca().set_yticks(range(0,length(to_show)-1))
	plt.gca().set_yticklabels(kets)
	plt.gcf()
end



begin
	proj, trunc_basis = trunc_subspace(es[to_show])
	const H_full = LazySum([1.0, 0.0], hermitify.([trunc_op(proj, Hzp)-identityoperator(trunc_op(proj, Hzp))*shift, trunc_op(proj, Hdrive)]))
end



begin
	envelope(t, tgate, sigma, drive_amp) = drive_amp*(exp(-(t-tgate/2)^2/(2*sigma^2))-exp(-(tgate/2)^2/(2*sigma^2)))
	drive(t, t_gate, ωd1, ωd2, amp1, amp2) = envelope(t,t_gate, t_gate/4, amp1)*cos(ωd1*t) + envelope(t, t_gate, t_gate/4, amp2)*cos(ωd2*t)
end



begin
	
	delta = 0.003*c
	w09 = ev[10]-ev[1] - delta
	w29 = ev[10]-ev[3] - delta
	scale = 2
	amp1 = abs(0.005*c/H_full.operators[2].data[3,1])/scale
	amp2 = abs(0.005*c/H_full.operators[2].data[3,2])/scale
	t_gate = 2000
	function Ht(t, psi)
		H_full.factors[2] = drive(t, t_gate, w09, w29, amp1, amp2)
		return H_full
	end
	states = [Ket(trunc_basis, one_vec(length(to_show), n)) for n in range(1,length(to_show))]
	w09, w29, amp1, amp2
end



begin
	times = range(0, t_gate, 1000)
	tout, psit = timeevolution.schroedinger_dynamic(times, states[1], Ht, maxiters=10^8)
	psit = [normalize(x) for x in psit]
	proj1 = tensor(states[1], dagger(states[1]))
	plt.close("all")
	for i in range(1,length(to_show))
		plt.plot(tout, real(expect(psi_to_proj(states[i]), psit)), label=L"|"*string(to_show[i]-1)*L"$\rangle$")
	end
	plt.xlabel("time (ns)")
	plt.ylabel("population")
	plt.legend()
	plt.gcf()
end



begin
	plt.close("all")
	plt.plot(tout, drive.(tout, t_gate, w09, w29, amp1, amp2))
	plt.xlabel("time (ns)")
	plt.ylabel("drive signal")
	plt.gcf()
end

##

using PRONTO
using StaticArrays

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end


function inprod(x)
    i = Int(length(x)/2)
    a = x[1:i]
    b = x[i+1:end]
    P = [a*a'+b*b' -(a*b'+b*a');
        a*b'+b*a' a*a'+b*b']
    return P
end

# H0 = round.(H_full.operators[1].data, digits=12)[1:3,1:3]
H0 = H_full.operators[1].data
# H0 = 2π*diagm([0, 3.4, 8.1, 9.1, 12.54])
H0 = diagm(diag(H0))
H0[1,1] = 0
H1 = H_full.operators[2].data
# H1 = real(H1)

function x_eig(i)
    w = eigvecs(collect(H0)) 
    x_eig = kron([1;0],real(w[:,i])) + kron([0;1],imag(w[:,i]))
end


# ------------------------------- 5lvl system to eigenstate 2 ------------------------------- ##

@kwdef struct lvl5 <: Model{10,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    P = I(10) - inprod(x_eig(2)) + inprod(x_eig(5))
	# P = inprod(x_eig(1)) + inprod(x_eig(4))
    1/2 * collect(x')*P*x
end


# ------------------------------- 5lvl system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    return mprod(-im*(H0+u[1]*H1))*x
end


stagecost(x,u,t,θ) = 1/2*θ.kl*collect(u')I*u + 0.0*collect(x')*mprod(diagm([0, 0, 0, 1, 0]))*x + 0.1*collect(x')*mprod(diagm([0, 0, 0, 0, 1]))*x

regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:5]
    x_im = x[6:10]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(5) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::lvl5) = SMatrix{10,10,Float64}(I(10) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(lvl5, dynamics, stagecost, termcost, regQ, regR)

## ------------------------------- demo: Simulation in 2000 ------------------------------- ##

x0 = SVector{10}(x_eig(3))

θ = lvl5(kl=0.01, kr=1, kq=1)

t0,tf = τ = (0,200)
tgate = tf

# μ = @closure t->SVector{1}(drive(t, t_gate, w09, w29, amp1, amp2))
μ = @closure t->SVector{1}(0.4*cos((H0[3,3]-H0[2,2])*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

##

ts = 0:0.01:tf
is = eachindex(ξ.x)
xs = [ξ.x(t)[i] for t∈ts, i∈is]
ps = ([I(5) I(5)] * (xs.^2)')'
us = [ξ.u(t) for t∈ts]

begin
	plt.close("all")
	plt.plot(ts,ps)
	plt.xlabel("time (ns)")
	plt.ylabel("population")
	plt.legend(["0", "2", "9","11","18"])
	plt.gcf()
end

##

using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt2_0pi_200T.mat", "w")
write(file, "Uopt2", us)
close(file)