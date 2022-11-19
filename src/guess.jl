# φ = guess_ol(M,θ,t0,tf,x0,ug) # where ug is a function of t
# φ = guess_zi(M,θ,t0,tf,xg) # where xg is a function of t
# xg = smooth(x0,xf,T) # returns a closure for a smooth function from x0->xf
# φ = guess_cl(M,θ,t0,tf,φg) # where φg is a function of t



function ol_ode(dξ,ξ,(M,θ),t)
    dx = @view dξ[1:nx(M)]
    du = @view dξ[(nx(M)+1):end]
    u = @view ξ[(nx(M)+1):end]

    f!(M,dx,θ,t,ξ)
    du .= .- u
end



# generate the open-loop trajectory of the system M with parameters θ from x0 at t0 until tf
function guess_ol(M::Model{NX,NU,NΘ},θ,t0,tf,x0,ug) where {NX,NU,NΘ}
    ODE(ol_ode, [x0;ug(t0)], (t0,tf), (M,θ), Buffer{Tuple{NX+NU}}(); dae=dae(M))
end



#= #TODO: finish

# generate the zero-input trajectory of the system M with parameters θ from t0 until tf
function guess_zi(M,θ,t0,tf,xg)
    # ug = t->0
    # solve ol
end


# generate a smooth curve from x0 to xf, then simulate a regulated zero-input response around it
function guess_φ()
    # make φg
    # build regulator
    # simulate projection
end


=#



#MAYBE:
# visualize(φ) # plot φ where φ is an ODE object or a closure (t)->(...)







# #MAYBE: convert to ξ0?
# function guess_zi(M::Model{NX,NU,NΘ},θ,x0,u0,t0,tf) where {NX,NU,NΘ}
#     ODE(ol_ode, [x0;u0], (t0,tf), (M,θ), Buffer{Tuple{NX+NU}}(); dae=dae(M))
# end


# export smooth, guess_φ

# smooth(t, x0, xf, T) = @. (xf - x0)*(tanh((2π/T)*t - π) + 1)/2 + x0

# # φg can be a closure, eg.
# # φg = @closure t->[smooth(t,x0,xf,T); 0.0]
# function guess_φ(M::Model{NX,NU,NΘ},θ,ξ0,t0,tf,φg) where {NX,NU,NΘ}
#     Pr_f = diagm(ones(NX))
#     Pr = ODE(Pr_ode, Pr_f, (tf,t0), (M,θ,φg), Buffer{Tuple{NX,NX}}())
#     # return Pr
#     φ = ODE(ξ_ode, ξ0, (t0,tf), (M,θ,φg,Pr), Buffer{Tuple{NX+NU}}(); dae=dae(M))
#     return φ
# end



# for validation
# run an OL dynamic response with μ provided by PRONTO
# plot a custom function on the resultant trajectory
