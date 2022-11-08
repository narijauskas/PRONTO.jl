
# @build Kr(M,θ,t,ξ,φ,Pr) -> inv($Rr)*(($Br)'*$Pr)
macro build(T, ex)

        
    #NOTE: to understand where this comes from, try:
    # ex = :(Kr(θ,t,ξ,φ,Pr) -> inv(Rr)*((Br)'*Pr))
    # dump(ex)
    # ex.args[1].args[2:end]


    # extract function name and arguments
    _fn = ex.args[1].args[1]
    _fn! = _!(_fn) # generates :(fn!)
    fn = esc(_fn)
    fn! = esc(_fn!)
    # first arg is name, second is M
    args = esc.(ex.args[1].args[3:end])
    def = ex.args[2]
    # T = esc(_T)

    return quote
        # symbolically generate local definitions for fn/fn!
        local $_fn, $_fn! = build($(args...)) do $(args...)
            $def
        end

        # define PRONTO.fn(M, args...) = local fn(args...)
        function ($fn)(M::$T, $(args...))
            ($_fn)($(args...))
        end

        # define PRONTO.fn!(M, buf, args...) = local fn!(buf, args...)
        function ($fn!)(M::$T, buf, $(args...))
            ($_fn!)(buf, $(args...))
        end
    end

end



function _derive(T)::Expr

    T = esc(T)
    M = :($T())
    Ar = :(fx($M,θ,t,φ))
    Br = :(fu($M,θ,t,φ))
    Pr = :(collect(Pr))
    Rr = :(Rr($M,θ,t,φ))
    Qr = :(Qr($M,θ,t,φ))
    Kr = :(Kr($M,θ,t,φ,Pr))
    user_Qr = esc(esc(:Qr))
    user_Rr = esc(esc(:Rr))
    μ = :(φ[(nx($M)+1):end])
    α = :(φ[1:nx($M)])
    x = :(ξ[1:nx($M)])
    u = :(ξ[(nx($M)+1):end])
    A = :(fx($M,θ,t,ξ))

    return quote

        @build $T Qr(M,θ,t,ξ) -> begin

            # local x,u = split(($T)(),ξ)
            ($user_Qr)(θ,t,$x,$u)
        end

        @build $T Rr(M,θ,t,ξ) -> begin

            # local x,u = split(($T)(),ξ)
            ($user_Rr)(θ,t,$x,$u)
        end

        # @build $T dPr_dt()
        @build $T Kr(M,θ,t,φ,Pr) -> inv($Rr)*(($Br)'*$Pr)
        @build $T Pr_t(M,θ,t,φ,Pr) -> riccati($Ar,$Kr,$Pr,$Qr,$Rr)


        @build $T ξ_t(θ,t,ξ,φ,Pr) -> begin
            # local x,u = split($T(),ξ)
            # local α,μ = split($T(),φ)
            vcat(

                $A...,
                $μ - Kr*($x-$α) - $u...
            )
        end
    end    
end



# # split(M::Model, ξ) = (ξ[1:nx(M)], ξ[(nx(M)+1):end])
# x = ξ[1:nx($M)]
# ξ[(nx(M)+1):end]

# function _projection(T)::Expr

#     #MAYBE: split here



#     return quote

        
#     end
# end





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

        iinfo("regulator solver ... "); @tick
        $(_regulator(T)); @tock; println(@clock)

        #=
        iinfo("projection solver ... "); @tick
        $(_projection(T)); @tock; println(@clock)
        
        iinfo("optimizer solver ... "); @tick
        $(_optimizer(T)); @tock; println(@clock)
        
        iinfo("lagrangian/costate solver ... "); @tick
        $(_lagrangian(T)); @tock; println(@clock)

        iinfo("search direction solver ... "); @tick
        $(_search_direction(T)); @tock; println(@clock)

        iinfo("cost derivative solver ... "); @tick
        $(_cost_derivatives(T)); @tock; println(@clock)
        
        iinfo("armijo rule ... "); @tick
        $(_armijo(T)); @tock; println(@clock)
        =#
        @tock derive_time
        info("model derivation completed in $(@clock derive_time)\n")
    end
end
#MAYBE: precompile model equations?
