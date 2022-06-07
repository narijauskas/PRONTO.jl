using Symbolics

# struct Model
#     # model functions...
#     f::Function
#     fx::Function
#     fu::Function
#     fxx::Function
#     fxu::Function
#     fuu::Function
#     l::Function
#     lx::Function
#     lu::Function
#     lxx::Function
#     lxu::Function
#     luu::Function
#     p::Function
#     px::Function
#     pxx::Function
#     # ... and their return types (or can we generate these?)
#     fT::DataType
#     fxT::DataType
#     fuT::DataType
#     fxxT::DataType
#     fxuT::DataType
#     fuuT::DataType
#     lT::DataType
#     lxT::DataType
#     luT::DataType
#     lxxT::DataType
#     lxuT::DataType
#     luuT::DataType
#     pT::DataType
#     pxT::DataType
#     pxxT::DataType
# end
# model = pronto_model(...)



struct FModel
    f::Function
    fT::DataType
end
# give symbolic x,u? generic x,u?










#f(x,u)
#l(x,u)
#p(x)
# where x is the code object representing the mathematical x(t)
# function build_model(f,l,p;NX,NU)
#     @variables x[1:NX] u[1:NU]
#     fx = jacobian(x, model.f, x, u),
#     fu = jacobian(u, model.f, x, u),
    
#     return model
# end

# model is a struct
model = (
    f = f,
    fx = jacobian(x,f,x,u)
)