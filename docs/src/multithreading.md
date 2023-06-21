# Multi-threading

PRONTO can use multiple threads to speed up parts of its computation.

To check if you are using multiple threads, run:
```julia
Threads.nthreads()
```

## Setup

To start julia with mulitple threads, it needs to be run with the flag `--threads auto`. Starting from a terminal, this looks like:
```
julia --threads auto
```

If you're using julia VS code integration, add `"julia.NumThreads": "auto"` to your `settings.json` file.

If using a symlink, you can add this argument

## Example

If you are using a large machine, and want to run parametric sweeps using PRONTO, it can be useful to do something like:

```julia
using ThreadTools

ξs = tmap(1:10) do k
    θ = MyModel(kq=k)
    pronto(θ,...)
end
```