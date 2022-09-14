
module PRONTO
abstract type Model{NX,NU} end

function f(T::Model,x,u,t)
    @error "function f undefined for models of type $(typeof(T))"
end

nx(M::Model{NX,NU}) where {NX,NU} = NX
nu(M::Model{NX,NU}) where {NX,NU} = NU

function pronto(M::Model{NX,NU}, args...) where {NX,NU}
end
# fallback if type is given:
pronto(T::DataType, args...) = pronto(T(), args...)


end # module





# define problem type:
struct FooSystem <: PRONTO.Model end
# struct FooSystem <: PRONTO.Model{NX,NU} end

# define: f,l,p, regulator

# autodiff/model setup
## ----------------------------------- @configure FooSystem ----------------------------------- ##
PRONTO.f(::FooSystem,x,u,t) = f(x,u,t)
@variables _x[1:model.NX] _u[1:model.NU]
# _x[1:nx(FooSystem())]
PRONTO.fx(::FooSystem,x,u,t) = jacobian(_x,f,_x,_u; inplace=false) # NX,NX #NOTE: for testing

# run pronto
pronto(FooSystem, α, μ, parameters...)