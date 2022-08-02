
# --------------------------------- regulator --------------------------------- #

#TODO: regulator(model,t,α,μ)
# function regulator(::Val{NX},::Val{NU},α,μ,model) where {NX,NU}
function regulator(α,μ,model)
    NX = model.NX; NU = model.NU; 
    T = model.T;
    Qr = model.Qr; Rr = model.Rr; iRr = model.iRr;

    fx! = model.fx!; _Ar = MMatrix{NX,NX,Float64}(undef)
    Ar = @closure (t)->(fx!(_Ar,α(t),μ(t)); return _Ar)
    fu! = model.fu!; _Br = MMatrix{NX,NU,Float64}(undef)
    Br = @closure (t)->(fu!(_Br,α(t),μ(t)); return _Br)
    
    # _Kr = MMatrix{NU,NX,Float64}(undef)
    _Kr = Buffer{Tuple{NU,NX}}()
    
    PT = collect(I(NX))
    Pr! = solve(ODEProblem(riccati!, PT, (T,0.0), (Ar,Br,Rr,Qr,_Kr)))
    _Pr = MMatrix{NX,NX,Float64}(undef)
    Pr = @closure (t)->(Pr!(_Pr, t); return _Pr)
    # Pr = functor((Pr,t) -> Pr!(Pr, t), MMatrix{NX,NX,Float64}(undef))
    
    iRrBr = MMatrix{NU,NX,Float64}(undef)
    function Kr(t)
        mul!(iRrBr, iRr(t), Br(t)')
        mul!(_Kr, iRrBr, Pr(t))
        return _Kr
    end
    return Kr
end

function riccati!(dP, P, (Ar,Br,Rr,Qr,Kr), t)
    mul!(Kr, Rr(t)\Br(t)', P)
    dP .= -Ar(t)'P - P*Ar(t) + Kr'*Rr(t)*Kr - Qr(t)
end
