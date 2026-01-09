function circle(n)
    R = zeros(Bool, n, n)
    c = div(n, 2) + 1
    for i in 1:n, j in 1:n
        R[i, j] = ((i - c)^2 + (j - c)^2) .<= (n - c)^2
    end
    return R
end

function open_reconstruct(Bf, B, R)
    S = Bf
    n = 0
    while n != count(S)
        n = count(S)
        S = dilate(S, R) .&& B
    end
    return S
end

smooth_mask(Bf, a) = exp.(-a * distance_transform(feature_transform(Bf)))
