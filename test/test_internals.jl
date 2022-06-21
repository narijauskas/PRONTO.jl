# goals: ensure type stability in example problems


# #TODO:
# using PRONTO


# include("test_lane_change.jl")


macro showarg(x)
    show(x)
    # ... remainder of macro, returning an expression
end





macro unpack(model)
    return esc(quote
        fx! = $(model).fx!
        fu! = $(model).fu!
    end)
end

model = (fx! = "fx!", fu! = "fu!")


function foo(model)
    @unpack model
    return fx!
end
