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

skipnan(x) = filter(!isnan, x)
skipnan(f::Function) = skipvalues(skipnan, f)
skipnanneg(x) = filter(x -> !isnan(x) && x > 0, x)
skipnanneg(f::Function) = skipvalues(skipnanneg, f)
function skipvalues(skiper::Function, f::Function)
    function g(x)
        x = skiper(x)
        if isempty(x)
            return NaN
        else
            return f(x)
        end
    end
    return g
end
