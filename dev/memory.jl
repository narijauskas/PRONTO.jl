Base.gc_live_bytes() # bytes tracked by GC

windows_rss() = parse(Int, readlines(`wmic process where processid=$(getpid()) get WorkingSetSize`)[2])







include("../examples/x_gate.jl")
θ = XGate3()
τ = t0,tf = 0,10
ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{12}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
pronto(θ, x0, η, τ; show_preview=false); # optimal trajectory



gc_bytes = Int[]
rss_bytes = Int[]
GC.gc()
Profile.take_heap_snapshot("before.heapsnapshot")
push!(gc_bytes, Base.gc_bytes())
push!(rss_bytes, windows_rss())

for ix in 1:10
    pronto(θ, x0, η, τ; show_preview=false); # optimal trajectory
    GC.gc()
    push!(gc_bytes, Base.gc_bytes())
    push!(rss_bytes, windows_rss())
end

Profile.take_heap_snapshot("after.heapsnapshot")
lineplot(gc_bytes; width=120)
lineplot(rss_bytes-gc_bytes; width=120)

# heap snapshot is missing close to 8GB of stuff



# problem is worse on linux:
# 16GB memory
# Intel Core i7-8700K CPU # 3.70GHz × 12
# Pop!_OS 22.04 LTS