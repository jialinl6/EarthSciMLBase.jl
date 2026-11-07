export MapAlgorithm, MapBroadcast, MapThreads, MapKernel, MapReactant, MapPolyester

"""
A type to specify the algorithm used for performing a computation
across a range of grid cells.
"""
abstract type MapAlgorithm end

"""
Perform computations by broadcasting.
"""
struct MapBroadcast <: MapAlgorithm end
"""
Perform computations in parallel using multi-threading.
"""
struct MapThreads <: MapAlgorithm end
"""
Perform computations on CPU or GPU using AcceleratedKernels.jl.

kwargs are passed to the AcceleratedKernels.jl `foreachindex` function.
"""
struct MapKernel <: MapAlgorithm
    kwargs::Dict{Symbol, Any}
    function MapKernel(; kwargs...)
        new(kwargs)
    end
end
"""
Compile and run computations using MLIR with Reactant.jl
"""
struct MapReactant <: MapAlgorithm end
"""
Perform computations on CPU threads with Polyester.jl's low-latency worker pool
(the `EarthSciMLBasePolyesterExt` extension; load Polyester to enable it).

Use it when the surrounding code already runs on Polyester (e.g. EarthSciAST's
generated kernels): its workers keep spinning after a batch, so work handed to
ordinary Julia tasks right afterwards waits for them and runs nearly serially.
Without Polyester loaded it falls back to the generic `MapAlgorithm` methods.
"""
struct MapPolyester <: MapAlgorithm end

function map_closure_to_range(f, range, ::MapAlgorithm = MapThreads(), args...; kwargs...)
    ThreadsX.map(range) do i
        f(i, args...; kwargs...)
    end
end
function map_closure_to_range(f, range, ::MapBroadcast, args...; kwargs...)
    f2(i) = f(i, args...; kwargs...)
    f2.(range)
end
function map_closure_to_range(f, range, mk::MapKernel, args...; kwargs...)
    bknd = if (length(args) > 0) && (args[1] isa AbstractArray)
        AK.get_backend(args[1])
    else
        error("No backend specified for MapKernel. Please provide an array as the first argument.")
    end
    AK.foreachindex(range, bknd; mk.kwargs...) do i
        f(i, args...; kwargs...)
    end
end
function mapreduce_range(f, op, range, ::MapAlgorithm, args...)
    bknd = if (length(args) > 0) && (args[1] isa AbstractArray)
        AK.get_backend(args[1])
    else
        error("No backend specified for MapKernel. Please provide an array as the first argument.")
    end
    f2(i) = f(i, args...)
    AK.mapreduce(f2, op, range, bknd; init = 0, neutral = 0)
end
