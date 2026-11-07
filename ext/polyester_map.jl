# MapPolyester's methods, run on Polyester's worker pool. Included by EarthSciMLBasePolyesterExt and, as a plain file,
# by code that includes EarthSciMLBase's source files directly (e.g. reseact.esm's blockdiag_local.jl); it extends
# whichever `map_closure_to_range` / `mapreduce_range` / `MapPolyester` and `Polyester` are in scope.

# Like the broadcast and ThreadsX methods, returns the per-index results (`+` and `mul!` of a BlockDiagonal use them),
# or `nothing` when `f` returns nothing (the block solve). Index 1 runs first, on the calling thread, to fix the result
# type; every index still runs exactly once.
# The range is `rng`, not `range`: `@batch` takes a name that is a constant global (`Base.range`) for that global and
# lets the loop body close over it, and a closure cannot become the `@cfunction` it runs on aarch64.
function map_closure_to_range(f, rng, ::MapPolyester, args...; kwargs...)
    isempty(rng) && return Any[]
    g(i) = f(i, args...; kwargs...)
    r1 = g(first(rng))
    if r1 === nothing
        Polyester.@batch for k in 2:length(rng)
            g(rng[k])
        end
        return nothing
    end
    out = Vector{typeof(r1)}(undef, length(rng))
    out[1] = r1
    Polyester.@batch for k in 2:length(rng)
        out[k] = g(rng[k])
    end
    return out
end

# Same contract as the generic method (an integer reduction from 0, as for the per-block LU infos).
function mapreduce_range(f, op, range, alg::MapPolyester, args...)
    return reduce(op, map_closure_to_range(f, range, alg, args...); init = 0)
end
