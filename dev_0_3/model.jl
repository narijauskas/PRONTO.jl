
# eg. @build Kr(M,θ,t,ξ,φ,Pr) -> inv($Rr)*(($Br)'*$Pr)
macro build(T, ex)

    # extract function name and arguments
    _fn = ex.args[1].args[1]
    _fn! = _!(_fn) # generates :(fn!)
    fn = esc(_fn)
    fn! = esc(_fn!)
    # first arg is name, second is M
    args = esc.(ex.args[1].args[3:end])
    def = ex.args[2]
 
    #NOTE: to understand where this comes from, try:
    # ex = :(Kr(θ,t,ξ,φ,Pr) -> inv(Rr)*((Br)'*Pr))
    # dump(ex)
    # ex.args[1].args[2:end]

    return quote
        # symbolically generate local definitions for fn/fn!
        local $_fn, $_fn! = build($(args...)) do $(args...)
            $def
        end

        # define PRONTO.fn(M, args...) = local_fn(args...)
        function ($fn)(M::$T, $(args...))
            ($_fn)($(args...))
        end

        # define PRONTO.fn!(M, buf, args...) = local_fn!(buf, args...)
        function ($fn!)(M::$T, buf, $(args...))
            ($_fn!)(buf, $(args...))
        end
    end

end



function _derive(T)::Expr

    T = esc(T)
    M = :($T())

    Ar = :(PRONTO.fx($M,θ,t,φ))
    Br = :(PRONTO.fu($M,θ,t,φ))
    Pr = :(collect(Pr))
    Rr = :(PRONTO.Rr($M,θ,t,φ))
    Qr = :(PRONTO.Qr($M,θ,t,φ))
    Kr = :(PRONTO.Kr($M,θ,t,φ,Pr))
    user_Qr = esc(esc(:Qr))
    user_Rr = esc(esc(:Rr))

    α = :(φ[1:nx($M)])
    μ = :(φ[(nx($M)+1):end])
    x = :(ξ[1:nx($M)])
    u = :(ξ[(nx($M)+1):end])
    z = :(ζ[1:nx($M)])
    v = :(ζ[(nx($M)+1):end])
    α̂ = :(φ̂[1:nx($M)])
    μ̂ = :(φ̂[(nx($M)+1):end])

    A = :(PRONTO.fx($M,θ,t,ξ))
    B = :(PRONTO.fu($M,θ,t,ξ))
    a = :(PRONTO.lx($M,θ,t,ξ))
    b = :(PRONTO.lu($M,θ,t,ξ))
    λ = :(collect(λ))

    Qo_1 = :(PRONTO.lxx($M,θ,t,ξ))
    Ro_1 = :(PRONTO.luu($M,θ,t,ξ))
    So_1 = :(PRONTO.lxu($M,θ,t,ξ))
    #TODO: create convenience access for these
    Qo_2 = :(PRONTO.lxx($M,θ,t,ξ) + sum(λ[k]*PRONTO.fxx($M,θ,t,ξ)[k,:,:] for k in 1:nx($M)))
    Ro_2 = :(PRONTO.luu($M,θ,t,ξ) + sum(λ[k]*PRONTO.fuu($M,θ,t,ξ)[k,:,:] for k in 1:nx($M)))
    So_2 = :(PRONTO.lxu($M,θ,t,ξ) + sum(λ[k]*PRONTO.fxu($M,θ,t,ξ)[k,:,:] for k in 1:nx($M)))
    Po = :(collect(Po))
    Ko_1 = :(PRONTO.Ko_1($M,θ,t,ξ,Po))
    Ko_2 = :(PRONTO.Ko_2($M,θ,t,ξ,λ,Po))
    ro = :(collect(ro))
    vo_1 = :(PRONTO.vo_1($M,θ,t,ξ,ro))
    vo_2 = :(PRONTO.vo_2($M,θ,t,ξ,λ,ro))


    return quote

        iinfo("regulator ... "); @tick
        @build $T Qr(M,θ,t,ξ) -> ($user_Qr)(θ,t,$x,$u)
        @build $T Rr(M,θ,t,ξ) -> ($user_Rr)(θ,t,$x,$u)
        @build $T Kr(M,θ,t,φ,Pr) -> inv($Rr)*(($Br)'*$Pr)
        @build $T dPr_dt(M,θ,t,φ,Pr) -> riccati($Ar,$Kr,$Pr,$Qr,$Rr)
        @tock; println(@clock)

        iinfo("projection ... "); @tick
        @build $T dξ_dt(M,θ,t,ξ,φ,Pr) -> vcat(

            PRONTO.f($M,θ,t,ξ)...,
            $μ - $Kr*($x-$α) - $u...
        )
        @tock; println(@clock)
        
        iinfo("lagrangian ... "); @tick
        @build $T dλ_dt(M,θ,t,ξ,φ,Pr,λ) -> -($A - $B*$Kr)'*$λ - $a + ($Kr)'*$b
        @tock; println(@clock)

        iinfo("1st order optimizer ... "); @tick
        @build $T Ko_1(M,θ,t,ξ,Po) -> inv($Ro_1)*(($So_1)' .+ ($B)'*$Po)
        @build $T dPo_dt_1(M,θ,t,ξ,Po) -> riccati($A,$Ko_1,$Po,$Qo_1,$Ro_1)
        @tock; println(@clock)

        iinfo("2nd order optimizer ... "); @tick
        @build $T Ko_2(M,θ,t,ξ,λ,Po) -> inv($Ro_2)*(($So_2)' .+ ($B)'*$Po)
        @build $T dPo_dt_2(M,θ,t,ξ,λ,Po) -> riccati($A,$Ko_2,$Po,$Qo_2,$Ro_2)
        @tock; println(@clock)

        iinfo("1st order costate ... "); @tick
        @build $T vo_1(M,θ,t,ξ,ro) -> inv(-$Ro_1)*(($B)'*$ro + $b)
        @build $T dro_dt_1(M,θ,t,ξ,Po,ro) -> -($A - $B*$Ko_1)'*$ro - $a + ($Ko_1)'*$b
        #  costate($A,$B,$a,$b,$Ko_1,$ro)
        @tock; println(@clock)

        iinfo("2nd order costate ... "); @tick
        @build $T vo_2(M,θ,t,ξ,λ,ro) -> inv(-$Ro_2)*(($B)'*$ro + $b)
        @build $T dro_dt_2(M,θ,t,ξ,λ,Po,ro) -> -($A - $B*$Ko_2)'*$ro - $a + ($Ko_2)'*$b
        # costate($A,$B,$a,$b,$Ko_2,$ro)
        @tock; println(@clock)

        iinfo("1st order search direction ... "); @tick
        @build $T dζ_dt_1(M,θ,t,ξ,ζ,Po,ro) -> vcat(
                
            $A*$z + $B*$v...,
            $vo_1 - $Ko_1*$z - $v...
        )
        @build $T _v(M,θ,t,ξ,ζ,Po,ro) -> $vo_1 - $Ko_1*$z
        @tock; println(@clock)

        iinfo("2nd order search direction ... "); @tick
        @build $T dζ_dt_2(M,θ,t,ξ,ζ,λ,Po,ro) -> vcat(
                
            $A*$z + $B*$v...,
            $vo_2 - $Ko_2*$z - $v...
        )
        @tock; println(@clock)

        #YO: can we solve these separately?
        iinfo("cost derivatives ... "); @tick
        @build $T dy_dt(M,θ,t,ξ,ζ,λ) -> vcat(

            ($a)'*($z) + ($b)'*($v),
            ($z)'*($Qo_2)*($z) + 2*($z)'*($So_2)*($v) + ($v)'*($Ro_2)*($v)
        )
        @build $T _Dh(M,θ,t,φ,ζ,y) -> y[1] + (PRONTO.px($M,θ,t,φ))'*($z)
        @build $T _D2g(M,θ,t,φ,ζ,y) -> y[2] + ($z)'*PRONTO.pxx($M,θ,t,φ)*($z)
        @tock; println(@clock)

        iinfo("armijo ... "); @tick
        @build $T dφ̂_dt(M,θ,t,ξ,φ,ζ,φ̂,γ,Pr) -> vcat(

            PRONTO.f($M,θ,t,φ̂)...,
            ($u + γ*$v) - ($Kr)*($α̂ - ($x + γ*$z)) - $μ̂...
        )
        @build $T dh_dt(M,θ,t,ξ) -> PRONTO.l($M,θ,t,ξ)
        @tock; println(@clock)
        
    end    
end







macro derive(T)

    # make sure we use the local context
    T = esc(T)

    return quote

        println()
        info("deriving the $(as_bold("$($T)")) model:")
        @tick derive_time

        # generate symbolic variables for derivation
        iinfo("preparing symbolics ... "); @tick
        $(_symbolics(T)); @tock; println(@clock)

        iinfo("dynamics derivatives ... "); @tick
        $(_dynamics(T)); @tock; println(@clock)

        iinfo("stage cost derivatives ... "); @tick
        $(_stage_cost(T)); @tock; println(@clock)
        
        iinfo("terminal cost derivatives ... "); @tick
        $(_terminal_cost(T)); @tock; println(@clock)

        $(_derive(T))
        
        @tock derive_time
        info("model derivation completed in $(@clock derive_time)\n")
    end
end
#MAYBE: precompile model equations?
