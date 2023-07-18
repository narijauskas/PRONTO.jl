using GLMakie

plot(rand(10))

using PRONTO: Data

Base.length(data::Data) = length(data.φ)




function 🙂(ax, data, i)
    γs = 0.0:0.05:1.2
    hs = [PRONTO.cost(data.ξ[i],τ)+data.Dh[i]*γ for γ in γs]
    h4s = [PRONTO.cost(data.ξ[i],τ)+data.Dh[i]*γ*0.4 for γ in γs]

    gs = map(γs) do γ
        η = PRONTO.armijo_projection(θ,x0,data.ξ[i],data.ζ[i],γ,data.Kr[i],τ)
        PRONTO.cost(η, τ)
    end

    h2s = map(γs) do γ
        PRONTO.cost(data.ξ[i],τ) + data.Dh[i]*γ + data.D2g[i]*γ^2/2
    end


    lines!(ax, γs, hs)
    lines!(ax, γs, gs)
    lines!(ax, γs, h2s)
    lines!(ax, γs, h4s)

    # display(fig)
    # return fig
    return nothing
end


fig = Figure()
ax = Axis(fig[1,1])
🙂(ax, data, 17)
ax = Axis(fig[2,1])
🙂(ax, data, 18)
display(fig)


n = ceil(Int,sqrt(length(data.φ)))

fig = Figure()
for i in 1:n, j in 1:n
    if (i-1)*n+j > length(data.φ)
        break
    end
    ax = Axis(fig[i,j])
    🙂(ax, data, (i-1)*n+j)
end
display(fig)

