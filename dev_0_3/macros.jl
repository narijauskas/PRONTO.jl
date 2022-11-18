using MacroTools
using MacroTools: postwalk

ex = quote
    NX = 3; NU = 2; NΘ = 1
end

ex2 = quote
    NX = 3
end


function dim(N, nex)
    y = 0
    postwalk(ex->(@capture(ex,$N=v_) && (y=v); ex) , nex)
    return y
end


3 == dim(:NX, ex)


macro model(name, ex)
    NX = dim(:NX, ex)
    NU = dim(:NU, ex)
    NΘ = dim(:NΘ, ex)

    quote
        struct $name <: PRONTO.Model{$NX,$NU,$NΘ}
        end

        let
            $ex
        end
    end
end

@macroexpand @model Foo begin
    NX = 3; NU = 2; NΘ = 1
end


using Symbolics

macro symbolics(T)
    return quote
        # create symbolic variables
        @variables θ[1:nθ($T())]
        @variables t
        @variables x[1:nx($T())] 
        @variables u[1:nu($T())]
        ξ = vcat(x,u)
        @variables α[1:nx($T())] 
        @variables μ[1:nu($T())]
        φ = vcat(α,μ)
        @variables z[1:nx($T())] 
        @variables v[1:nu($T())]
        ζ = vcat(z,v)
        @variables α̂[1:nx($T())] 
        @variables μ̂[1:nu($T())]
        φ̂ = vcat(α̂,μ̂)
        @variables Pr[1:nx($T()),1:nx($T())]
        @variables Po[1:nx($T()),1:nx($T())]
        @variables ro[1:nx($T())]
        @variables λ[1:nx($T())]
        @variables γ
        @variables y[1:2] #YO: can we separate these into scalar Dh/D2g?
        @variables h #MAYBE: rename j or J?
    end
end




if @capture(ex, f(θ,t,x,u) = user_f_)

    f_sym = collect(Base.invokelatest(eval(ex), args...)) # args are symbolics

    # load function definition expression
    # evaluate & symbolically derive
    # add definitions to pronto
