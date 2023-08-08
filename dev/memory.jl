using PRONTO #v1.0.0 on windows 11

windows_rss() = parse(Int, readlines(`wmic process where processid=$(getpid()) get WorkingSetSize`)[2])

include("../examples/x_gate.jl")
pronto(θ, x0, η, τ; show_preview=false); # run pronto

gc_bytes = Int[]
rss_bytes = Int[]
GC.gc()
Profile.take_heap_snapshot("before.heapsnapshot")
push!(gc_bytes, Base.gc_bytes())
push!(rss_bytes, windows_rss())

for ix in 1:10
    pronto(θ, x0, η, τ; show_preview=false); # run pronto
    GC.gc()
    push!(gc_bytes, Base.gc_bytes())
    push!(rss_bytes, windows_rss())
end

Profile.take_heap_snapshot("after.heapsnapshot")
lineplot(gc_bytes; width=120)
lineplot(rss_bytes-gc_bytes; width=120)

# heap snapshot is missing close to 8GB of stuff
