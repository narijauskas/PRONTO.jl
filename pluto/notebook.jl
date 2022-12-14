### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ f7a20b4f-b838-4cfc-b298-8a3d156e5675
import Pkg; Pkg.activate()

# ╔═╡ e1150180-7b4a-11ed-3706-43d9fdc0f59a
using WGLMakie, JSServe

# ╔═╡ ed580dc5-b200-44e5-b072-73912d7cc8dd
html"""<style>
main {
    max-width: 1200px;
}
"""

# ╔═╡ 04dc4811-de4d-438a-8106-b916143593a7
md"""
## Subtitle
- a
- b

What about math?

$$y = mx+b$$
"""

# ╔═╡ 8e562a0e-191a-4e66-ad09-3989917a3d1a
plot(rand(10))

# ╔═╡ 82da3532-382e-4f75-a72f-a9032fb38411
# model parameters
begin
    M = 2041    # [kg]     Vehicle mass
    J = 4964    # [kg m^2] Vehicle inertia (yaw)
    g = 9.81    # [m/s^2]  Gravity acceleration
    Lf = 1.56   # [m]      CG distance, front
    Lr = 1.64   # [m]      CG distance, back
    μ = 0.8     # []       Coefficient of friction
    b = 12      # []       Tire parameter (Pacejka model)
    c = 1.285   # []       Tire parameter (Pacejka model)
    s = 30      # [m/s]    Vehicle speed
end

# ╔═╡ ed66738b-ea83-450c-ad09-ea7a08e2fae1
begin
    # sideslip angles
    αf(x) = x[5] - atan((x[2] + Lf*x[4])/s)
    αr(x) = x[6] - atan((x[2] - Lr*x[4])/s)

    # tire force
    F(α) = μ*g*M*sin(c*atan(b*α))
end

# ╔═╡ 46b04c49-b0f6-41fc-8db0-5c98c8452c4e


# ╔═╡ Cell order:
# ╠═f7a20b4f-b838-4cfc-b298-8a3d156e5675
# ╠═e1150180-7b4a-11ed-3706-43d9fdc0f59a
# ╠═ed580dc5-b200-44e5-b072-73912d7cc8dd
# ╟─04dc4811-de4d-438a-8106-b916143593a7
# ╠═8e562a0e-191a-4e66-ad09-3989917a3d1a
# ╠═82da3532-382e-4f75-a72f-a9032fb38411
# ╠═ed66738b-ea83-450c-ad09-ea7a08e2fae1
# ╠═46b04c49-b0f6-41fc-8db0-5c98c8452c4e
