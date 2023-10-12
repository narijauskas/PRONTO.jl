var documenterSearchIndex = {"docs":
[{"location":"cheat_sheet/#Cheat-Sheet","page":"Cheat Sheet","title":"Cheat Sheet","text":"","category":"section"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Updated and adapted to match this implementation a bit more. Everything acts under a parameter set θ.","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"PRONTO iterates ξ, φ is the reference trajectory","category":"page"},{"location":"cheat_sheet/#Requirements","page":"Cheat Sheet","title":"Requirements","text":"","category":"section"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Dynamics: $ f(x(t),u(t),t,θ) $","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Stage Cost $ l(x(t),u(t),t,θ) $","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Terminal Cost $ p(x(t),u(t),t,θ) $","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Regulator Matrices $ R(x(t),u(t),t,θ), Q(x(t),u(t),t,θ) $","category":"page"},{"location":"cheat_sheet/#PRONTO","page":"Cheat Sheet","title":"PRONTO","text":"","category":"section"},{"location":"cheat_sheet/#Regulator","page":"Cheat Sheet","title":"Regulator","text":"","category":"section"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"The regulator is computed by:","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"K_r(t) = R_r(t)^-1 B_r(t)^T P_r(t)","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Defined by evaluating along α(t),μ(t):","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"A_r(t) = f_x(α(t)μ(t)tθ)","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"B_r(t) = f_u(α(t)μ(t)tθ)","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"R_r(t) = R(α(t)μ(t)tθ)","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Q_r(t) = Q(α(t)μ(t)tθ)","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Where P_r is found  by solving a differential riccati equation backwards in time from P(α(T)μ(T)T).","category":"page"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"-dotP_r = A_r^T P_r + P_r A_r - K_r^T R_r K_r + Q_r","category":"page"},{"location":"cheat_sheet/#Optimizer","page":"Cheat Sheet","title":"Optimizer","text":"","category":"section"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"K_o(t) = R_o^-1(S_o^T + B^T P)","category":"page"},{"location":"cheat_sheet/#Second-Order","page":"Cheat Sheet","title":"Second Order","text":"","category":"section"},{"location":"cheat_sheet/","page":"Cheat Sheet","title":"Cheat Sheet","text":"Q_o = L_xx = l_xx + sum λ_k f_kxx","category":"page"},{"location":"#PRONTO.jl","page":"PRONTO.jl","title":"PRONTO.jl","text":"","category":"section"},{"location":"","page":"PRONTO.jl","title":"PRONTO.jl","text":"Hello and welcome to the julia implementation of the PRojection-Operator-Based Newton’s Method for Trajectory Optimization (PRONTO).","category":"page"},{"location":"","page":"PRONTO.jl","title":"PRONTO.jl","text":"For now, please see the README - actual docs are coming soon! ","category":"page"},{"location":"devnotes/#PRONTO-Devnotes","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"","category":"section"},{"location":"devnotes/#.-Model-Definition","page":"PRONTO Devnotes","title":"1. Model Definition","text":"","category":"section"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"The model requires: $ f(θ,t,x,u) $","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"using PRONTO\n\n@model TwoSpin begin\n    NX = 4; NU = 1; NΘ = 0\n\n    f(θ,t,x,u) = ...\nend","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"This defines a type:","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"TwoSpin <: PRONTO.Model{4,1,0}","category":"page"},{"location":"devnotes/#.-Model-Derivation","page":"PRONTO Devnotes","title":"2. Model Derivation","text":"","category":"section"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"Symbolically generate functions for:","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"\nf(θtξ)\n\n\ndefined within the pronto module as\n\njulia\nPRONTOf(Mθtξ)\nPRONTOf(Mbufθtξ)\n\n\ndispatch on M where M  PRONTOModel\n\nMacros do the heavy lifting\n\n\nJacobian defined as\n\nJ_x (f) = beginbmatrix\nfracδf_δx_  cdots  fracδf_δx_\nvdots  ddots  vdots\nfracδf_δx_  cdots  fracδf_δx_\nendbmatrix","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"f_x = J_x(f) f_xu = J_u(J_x(f))","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"f2 = f_2","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"f_x21 = fracδf_2δx_1","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"f_xu213 = fracδfracδf_2δx_1δu_3","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"vdots","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"f[2] = f_2 # NX\nfx[2,1] = df2/dx1 # NX,NX\nfxu[2,1,3] = d(df2/dx1)/du3# NX,NX,NU","category":"page"},{"location":"devnotes/#Hello-Jay","page":"PRONTO Devnotes","title":"Hello Jay","text":"","category":"section"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"Here is a model:","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"\n@model Split begin\n    using LinearAlgebra\n        \n    NX = 22; NU = 1; NΘ = 0\n\n    function mprod(x)\n        Re = I(2)  \n        Im = [0 -1;\n            1 0]   \n        M = kron(Re,real(x)) + kron(Im,imag(x));\n        return M   \n    end\n\n    function inprod(x)\n        a = x[1:Int(NX/2)]\n        b = x[(Int(NX/2)+1):(2*Int(NX/2))]\n        P = [a*a'+b*b' -(a*b'+b*a');\n            a*b'+b*a' a*a'+b*b']\n        return P\n    end\n\n    N = 5\n    n = 2*N+1\n    α = 10\n    ω = 0.5\n\n    H = zeros(n,n)\n    for i = 1:n\n        H[i,i] = 4*(i-N-1)^2\n    end\n    v = -α/4 * ones(n-1)\n\n    H0 = H + Bidiagonal(zeros(n), v, :U) + Bidiagonal(zeros(n), v, :L)\n    H1 = Bidiagonal(zeros(n), -v*1im, :U) + Bidiagonal(zeros(n), v*1im, :L)\n    H2 = Bidiagonal(zeros(n), -v, :U) + Bidiagonal(zeros(n), -v, :L)\n\n    nu = eigvecs(H0)\n\n    nu1 = nu[:,1]\n    nu2 = nu[:,2]\n\n    f(θ,t,x,u) = mprod(-1im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2))*x\n    \n    Ql = zeros(2*n,2*n)\n    Rl = I\n    l(θ,t,x,u) = 1/2*x'*Ql*x + 1/2*u'*Rl*u\n    \n    Rr(θ,t,x,u) = diagm(ones(1))\n    Qr(θ,t,x,u) = diagm(ones(2*n))\n    \n    xf = [nu2;0*nu2]\n    function p(θ,t,x,u)\n        P = I(2*n) - inprod(xf)\n        1/2*x'*P*x\n    end\n\n\nend\n","category":"page"},{"location":"devnotes/#.-Intermediate-Operators","page":"PRONTO Devnotes","title":"3. Intermediate Operators","text":"","category":"section"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"Provide convenience for efficiently defining diffeq's later on. Eg.","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"K_r = R_r^-1 B_r^T P_r","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"Ar(M,θ,t,φ) = (fx!(M,buf,θ,t,φ); return buf)\nKr(M,θ,t,φ,Pr) = Rr(...)\\(Br(...)'Pr(...))","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"warning: yo\nshould these be in-place?\nif so, where should buffers be defined?\nthis behavior would almost be closer to an ode solution","category":"page"},{"location":"devnotes/#.-DiffEQs","page":"PRONTO Devnotes","title":"4. DiffEQs","text":"","category":"section"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"fracdPrdt = -A_r^T P_r - P_r A_r + K_r^T R_r K_r - Q_r","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"define an in-place version using the format of DifferentialEquations.jl","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"dPr_dt(dPr, Pr, (M,θ,φ), t)","category":"page"},{"location":"devnotes/#.-ODE-Solutions","page":"PRONTO Devnotes","title":"5. ODE Solutions","text":"","category":"section"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"uses internal buffer and is wrapped by FunctionWrappers.jl - todo: make thread-safe","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"Pr(t)","category":"page"},{"location":"devnotes/#.-Trajectories","page":"PRONTO Devnotes","title":"6. Trajectories","text":"","category":"section"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"special ODE solutions with pretty plots :)","category":"page"},{"location":"devnotes/#.-Buffered-Operators","page":"PRONTO Devnotes","title":"0. Buffered Operators","text":"","category":"section"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"ODE solutions, trajectories, and intermediate operators all behave the same way:","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"apply function in-place using internal buffer (with clear thread-safety)\nreturn a copy of the internal buffer (can use fancy types, eg. sparse arrays)","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"With some differences:","category":"page"},{"location":"devnotes/","page":"PRONTO Devnotes","title":"PRONTO Devnotes","text":"ODEs/Trajectories render as plots (why not everything?)\nintermediate operators and diffeqs should both have symbolic rendering (maybe a macro facilitates this, eg. @symbolic TwoSpin fx)","category":"page"}]
}
