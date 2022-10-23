# goals: ensure type stability in example problems


# #TODO:
# using PRONTO


# include("test_lane_change.jl")


ts->[-pi,pi]


## ----------------------------- scaling ----------------------------- ##

"""
rescale(x, a, b, c, d)
rescales a vector x ∈ [a,b] to x ∈ [c,d]
"""
function rescale(x, a, b, c, d)
    return (x .- a) .* ((d-c)/(b-a)) .+ c
end



rescale(x,c,d) = rescale(x, minimum(x), maximum(x), c, d)

normalize(x) = rescale(x, minimum(x), maximum(x), 0, 1)





# Xi = Interpolant()
fig = Figure()

ax = Axis(fig[1,1])
for (x,t) in splitseries(X_x)
    lines!(ax,t,x)
end

ax = Axis(fig[2,1])
for (u,t) in splitseries(U_u)
    lines!(ax,t,u)
end
display(fig)







fig = Figure()

ax = Axis(fig[1,1])
for (x,t) in splitseries(X_α)
    lines!(ax,t,x)
end

ax = Axis(fig[2,1])
for (u,t) in splitseries(U_μ)
    lines!(ax,t,u)
end
display(fig)



Kr = Interpolant(0:0.001:T, NU, NX)

@time foo!(Kr, fu!, X_α, U_μ, Pr)

function foo!(Kr, fu!, X_α, U_μ)
    Pr = solve(ODEProblem(riccati!, PT, (T,0.0), (fx!,fu!,Qr,Rr,iRr,X_α,U_μ)), Tsit5())


    for (Kr_t, t) in zip(Kr, times(Kr))
        Br = MArray{Tuple{NX,NU},Float64}(undef)
        iRrBr = MArray{Tuple{NU,NX},Float64}(undef)

        fu!(Br, X_α(t), U_μ(t))
        mul!(iRrBr, iRr(t), Br') # {NU,NU}*{NU,NX}->{NU,NX}
        mul!(Kr_t, iRrBr, Pr(t))
    end
    return Kr, Pr
end


tx = @elapsed update_Kr!(Kr,Pr,Pr_ode,fx!,fu!,iRr,Qr,X_α,U_μ)

    # Xi = Interpolant()
    fig = Figure()

    ax = Axis(fig[1,1])
    for (x,t) in splitseries(Kr)
        lines!(ax,t,x)
    end

    ax = Axis(fig[2,1])
    for (x,t) in splitseries(Pr)
        lines!(ax,t,x)
    end

    display(fig)
