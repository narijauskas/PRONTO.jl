include("TwoSpin.jl")
## ----------------------------------- run optimization ----------------------------------- ##

θ = TwoSpin()
τ = t0,tf = 0,10

x0 = @SVector [0.0, 1.0, 0.0, 0.0]
xf = @SVector [1.0, 0.0, 0.0, 0.0]
u0 = 0.1
μ = @closure t->SizedVector{1}(u0)
φ = open_loop(θ, xf, μ, τ) # guess trajectory
ξ = pronto(θ, x0, φ, τ) # optimal trajectory
@time ξ = pronto(θ, x0, φ, τ) # optimal trajectory
# @code_warntype PRONTO.f(x0,u0,t0,θ)




## ----------------------------------- symbolic ----------------------------------- ##
using Symbolics
using PRONTO: symbolic

# first, create symbolic versions of everything
θ = symbolic(TwoSpin)
θ = symbolic(LaneChange)
λ = Symbolics.variables(:λ, 1:NX)
x = Symbolics.variables(:x, 1:NX)
u = Symbolics.variables(:u, 1:NU)
t = symbolic(:t)

# symbolic(TwoSpin, PRONTO.f)

# can now symbolically trace any PRONTO kernel function by calling it, eg.
PRONTO.f(θ,x,u,t) # returns a symbolic representation of dynamics
PRONTO.fx(x,u,t,θ) # or dynamics jacobian
PRONTO.Lxu(λ,x,u,t,θ) # or lagrangian hessian 



# this is useful, especially combined with other packages
using Latexify
copy_to_clipboard(true)
latexify(PRONTO.f(x,u,t,θ))

## ----------------------------------- more symbolic ----------------------------------- ##

kernels = (:Q,:R,:f,:fx,:fu,:fxu,:fuu,:l,:lx,:lu,:lxx,:lxu,:luu,:p,:px,:pxx)


open("temp.tex", "w") do file
    for fn in (:Q,:R,:f,:fx,:fu,:fxu,:fuu,:l,:lx,:lu,:lxx,:lxu,:luu,:p,:px,:pxx)
        write(file, "##", fn, "\n")
        ex = getproperty(PRONTO, fn)(x,u,t,θ)
        write(file, latexify(ex), "\n\n")
    end

    for fn in (:Lxx, :Lxu, :Luu)
        write(file, "##", fn, "\n")
        ex = getproperty(PRONTO, fn)(λ,x,u,t,θ)
        write(file, latexify(ex), "\n\n")
    end
end

PRONTO.fx(x,u,t,θ)
PRONTO.fu(x,u,t,θ)
latexify(PRONTO.fxx(x,u,t,θ))
PRONTO.fuu(x,u,t,θ)
PRONTO.l(x,u,t,θ)
PRONTO.lx(x,u,t,θ)
PRONTO.lu(x,u,t,θ)

PRONTO.Q(x,u,t,θ)

PRONTO.Lxx(λ,x,u,t,θ)
PRONTO.Lxu(λ,x,u,t,θ)
PRONTO.Luu(λ,x,u,t,θ)