function to_nt(x::Dict)
    NamedTuple(k => x[k] for k in keys(x))
end
function to_dict(x::NamedTuple)
    Dict(k => x[k] for k in keys(x))
end
function to_dict(x)
    Dict(k => x[:, k] for k in propertynames(x))
end

sqrtz(x) = isapprox(x,0.0,atol=1e-10) ? 0 : sqrt(x)

function running_func(x::AbstractArray{<:Real}, n::Integer, f; step = 1)
    N = length(x)
    S = 1:step:N
    L = length(S)
    out = Array{Float64}(undef, L)
    for (j, i) in enumerate(S)
        l1 = max(1, i - div(n, 2))
        l2 = min(N, (i + div(n, 2)))
        r = l1:l2
        out[j] = f(view(x, r))
    end
    return out
end

function find_error_regions(F)
    s = Int64[]
    e = Int64[]
    i = 1
    L = length(F)
    while !isnothing(i)
        i = findnext(F, i)
        if isnothing(i)
            break
        else
            push!(s, i)
        end
        j = findnext(!, F, i)
        sj = isnothing(j) ? L : j - 1
        push!(e, sj)
        i = j # End loop if j==nothing
    end
    return (s, e)
end

import Base:
    parent,
    iterate,
    eltype,
    keys,
    eachindex,
    getindex,
    show,
    @propagate_inbounds,
    parent,
    IndexStyle,
    IteratorSize,
    IteratorEltype,
    SizeUnknown

# Sensors Errors are replaced with NaN values, they are skipped when applying statistics.
# The following code is directly adapted from Statistics.skipmissing
# However, this skipnan implementation allocates too much so we rely on NanStatistics instead. We keep it as a pratical tool in some cases.

skipnan(itr) = SkipNan(itr)

struct SkipNan{T}
    x::T
end
IteratorSize(::Type{<:SkipNan}) = SizeUnknown()
IteratorEltype(::Type{SkipNan{T}}) where {T} = IteratorEltype(T)
eltype(::Type{SkipNan{T}}) where {T} = eltype(T)
parent(itr::SkipNan) = itr.x

function iterate(itr::SkipNan, state...)
    y = iterate(itr.x, state...)
    y === nothing && return nothing
    item, state = y
    while isnan(item)
        y = iterate(itr.x, state)
        y === nothing && return nothing
        item, state = y
    end
    item, state
end

IndexStyle(::Type{<:SkipNan{T}}) where {T} = IndexStyle(T)
eachindex(itr::SkipNan) =
    Iterators.filter(i -> !isnan(@inbounds(itr.x[i])), eachindex(itr.x))
keys(itr::SkipNan) = Iterators.filter(i -> !isnan(@inbounds(itr.x[i])), keys(itr.x))
@propagate_inbounds function getindex(itr::SkipNan, I...)
    v = itr.x[I...]
    isnan(v) && throw(error(LazyString("the value at index ", I, " is NaN")))
    v
end

function show(io::IO, s::SkipNan)
    print(io, "skipfunc(")
    show(io, s.x)
    print(io, ')')
end
