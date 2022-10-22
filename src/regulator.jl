
# --------------------------------- regulator  η -> Kr --------------------------------- #

#TODO: regulator(model,t,α,μ)
# function regulator(::Val{NX},::Val{NU},α,μ,model) where {NX,NU}
# (::M)(α,μ) = (M.f!(...);...)
# (::M)(t) = M(M.α(t), M.μ(t))


function regulator(M::Model{NX,NU},α,μ) where {NX,NU}
    T = ?
    Pr! = solve(ODEProblem(riccati!, collect(1.0*I(NX)), (T,0.0), (Ar,Br,Rr,Qr)))
    
end


function riccati!(dP, P, (α,μ), t)
    Ar = Buffer{Tuple{NX,NX}}()
    fx!(Ar,α(t),μ(t))

    Br = Buffer{Tuple{NX,NU}}()
    fu!(Br,α(t),μ(t))

    # mul!(Kr, Rr(t)\Br(t)', P)
    Kr = Rr(t)\Br'*P
    dP .= -Ar'*P - P*Ar + Kr'*Rr(t)*Kr - Qr(t)
    #NOTE: dP is symmetric, as should be P
end



Kr = Buffer{Tuple{NU,NX}}()
Kr = Rr(t)\Br(t)'*P

# function + dimensions + trajectory -> closure + buffer
Ar[]
Ar!(α,μ,t)
function Ar(M::Model{NX,NU},α,μ,t) where {NX,NU}
    Ar = Buffer{Tuple{NX,NX}}()
    # Ar = @closure (t)->(fx!(_Ar,α(t),μ(t)); return _Ar)
    fx!(Ar,α(t),μ(t))
    return Ar
end

struct Buffer{S,F}
    f::F
    X::MArray{S,Float64}
end

X[] = get buffer
X(...) = update buffer & return


struct ModelBuffer{NX,NU}
    Ar::Buffer{Tuple{NX,NX}}
    # ...
end
# Ar = buffer(fx!, nx, nx, α, μ)
# I'm already creating new buffers each run...

function regulator(α,μ,model)
    NX = model.NX; NU = model.NU; 
    T = model.T;
    Qr = model.Qr; Rr = model.Rr; #iRr = model.iRr;

    fx! = model.fx!; _Ar = Buffer{Tuple{NX,NX}}()
    Ar = @closure (t)->(fx!(_Ar,α(t),μ(t)); return _Ar)
    fu! = model.fu!; _Br = Buffer{Tuple{NX,NU}}()
    Br = @closure (t)->(fu!(_Br,α(t),μ(t)); return _Br)
    
    _Kr = Buffer{Tuple{NU,NX}}()
    
    Pr! = solve(ODEProblem(riccati!, collect(1.0*I(NX)), (T,0.0), (Ar,Br,Rr,Qr)))
    _Pr = Buffer{Tuple{NX,NX}}()
    Pr = @closure (t)->(Pr!(_Pr, t); return _Pr)
        
    iRrBr = Buffer{Tuple{NU,NX}}()
    function Kr(t)
        # mul!(iRrBr, iRr(t), Br(t)')
        mul!(_Kr, Rr(t)\Br(t)', Pr(t))
        return _Kr
    end
    # Kr(t) = Kr(α(t), μ(t),t)
    return Kr
end

function riccati!(dP, P, (Ar,Br,Rr,Qr), t)
    # mul!(Kr, Rr(t)\Br(t)', P)
    Kr = Rr(t)\Br(t)'*P
    dP .= -Ar(t)'*P - P*Ar(t) + Kr'*Rr(t)*Kr - Qr(t)
    #NOTE: dP is symmetric, as should be P
end

