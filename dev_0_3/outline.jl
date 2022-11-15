
# part 1. define model

# part 2. autodiff model symbolically
# load functions into pronto
# facilitated by macro


# part 3. intermediate functions
# generic, pass model
# buffer? in-place? 
A(M,θ,t,ξ) = fx!(M,_A)

# part 4. diffeqs - inplace (buffer provided)

# part 5. ode solutions - wrapped by functionwrapper & buffer