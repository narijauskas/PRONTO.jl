# https://discourse.julialang.org/t/inverse-with-destination/2267/11
using LinearAlgebra
LinearAlgebra.inv!(lu!(A)) # general
LinearAlgebra.inv!(choelsky!(A)) # if SPD