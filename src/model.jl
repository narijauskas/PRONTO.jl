using Symbolics


# model = pronto_model(...)

# give symbolic x,u
function build_model(x,u,f,l,p)

    return model
end


model = (
    f = f,
    fx = jacobian(x,f,x,u)
)