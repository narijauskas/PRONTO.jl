# from: https://discourse.julialang.org/t/how-to-track-total-memory-usage-of-julia-process-over-time/91167/6
using Printf: @printf

function meminfo_julia()
  # @printf "GC total:  %9.3f MiB\n" Base.gc_total_bytes(Base.gc_num())/2^20
  # Total bytes (above) usually underreports, thus I suggest using live bytes (below)
  @printf "GC live:   %9.3f MiB\n" Base.gc_live_bytes()/2^20
  @printf "JIT:       %9.3f MiB\n" Base.jit_total_bytes()/2^20
  @printf "Max. RSS:  %9.3f MiB\n" Sys.maxrss()/2^20
end