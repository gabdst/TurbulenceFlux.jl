"""
    RepeatedMedianRegressor

A struct for the Repeated Median Regression algorithm.
"""
struct RepeatedMedianRegressor
    a::Matrix{Float64} # Slopes
    b::Vector{Float64} # Median of slopes
    c::Vector{Float64} # Centered signal
    m::Vector{Float64} # Trend
    s::Vector{Float64} # absolute deviations
    params::Vector{Float64} # params found
    d::Float64 # correction constant for absolute deviation
    n::Int64 # Size of the block
    function RepeatedMedianRegressor(n::Int64, d::Float64 = 2.2219)
        a = zeros(n, n)
        b = zeros(n)
        s = zeros(div(n * (n - 1), 2))
        c = zeros(n)
        params = zeros(3)
        m = zeros(n)
        return new(a, b, c, m, s, params, d, n)
    end
end

"""
    getparams(rmr::RepeatedMedianRegressor)

Return the estimated parameters (mu, beta, sigma) from the Repeated Median Regression.
"""
getparams(rmr::RepeatedMedianRegressor) =
    (; mu = rmr.params[1], beta = rmr.params[2], sigma = rmr.params[3])

"""

    (rmr::RepeatedMedianRegressor)(y::AbstractArray{<:Real})
Apply the Repeated Median Regression algorithm to a given time series `y`.

# Arguments
- `y::AbstractArray{<:Real}`: The time series data for which to perform regression.

# Returns
A tuple containing:
- `m::Vector{Float64}`: The trend component of the regression.
- `params::Tuple{Float64, Float64, Float64}`:  The estimated parameters (mu, beta, sigma).
"""
function (rmr::RepeatedMedianRegressor)(y::AbstractArray{<:Real})
    n = rmr.n
    # Some computations could be avoided, however this is apparently better for the compiler
    @turbo for j in axes(rmr.a, 2)
        for i in axes(rmr.a, 1)
            t_i = 2 * (i - 1) / (n - 1) - 1
            t_j = 2 * (j - 1) / (n - 1) - 1
            rmr.a[i, j] = (y[i] - y[j]) / (t_i - t_j)
        end
    end
    rmr.b .= vec(nanmedian!(rmr.a, dims = 1))
    beta_rm = nanmedian!(rmr.b)
    @turbo for i in axes(y, 1)
        t_i = 2 * (i - 1) / (n - 1) - 1
        rmr.c[i] = y[i] - t_i * beta_rm
    end
    # Linear to Upper Triangular matrix (no diagonal, thus (n*(n-1)/2) points)
    @turbo for k = 1:length(rmr.s)
        i = n - 1 - floor(Int, sqrt(-8 * k + 4 * n * (n - 1) + 1) / 2 - 0.5)
        j = k + i + ((n - i + 1) * (n - i) - n * (n - 1)) ÷ 2
        rmr.s[k] = abs(rmr.c[i] - rmr.c[j])
    end
    mu_rm = nanmedian!(rmr.c)
    sigma_rm = rmr.d * nanquantile!(rmr.s, 1 / 4)
    # Compute the mean signal
    @turbo for i = 1:n
        t_i = 2 * (i - 1) / (n - 1) - 1
        rmr.m[i] = mu_rm + t_i * beta_rm
    end
    rmr.params[1] = mu_rm
    rmr.params[2] = beta_rm
    rmr.params[3] = sigma_rm
    return rmr.m, getparams(rmr)
end

"""
    flag_spikes(x::AbstractArray{<:Real}, n::Int64, f = 5)

Flag potential spikes in a time series `x` using the Repeated Median Regression algorithm.
# Arguments
- `x::AbstractArray{<:Real}`: The time series data for which to identify spikes.
- `n::Int64`: The block size for the Repeated Median Regression.
- `f::Float64 = 5`: The scaling factor for the median absolute deviation.

# Returns
- `m::Vector{Bool}`: A boolean vector indicating the presence of spikes at each time point in `x`.  `true` indicates a spike.
"""
function flag_spikes(x::AbstractArray{<:Real}, n::Int64, f = 5)
    m = falses(length(x))
    rmr = RepeatedMedianRegressor(n)
    itr_x = Iterators.partition(x, n)
    itr_m = Iterators.partition(m, n)
    for (xv, mv) in zip(itr_x, itr_m)
        mu, params = rmr(xv)
        mv .= abs.(xv .- mu) .> params.sigma * f
    end
    return m
end


"""
    flag_nan(x::AbstractArray{<:Real})

Flag NaN values in an array `x` and return the corresponding Boolean vector.
"""
flag_nan(x::AbstractArray{<:Real}) = isnan.(x)

"""
    interpolate_errors!(x::AbstractArray{<:Real}, m::AbstractVector{Bool})

Interpolates over the flagged errors in `x` using the inverted mask `m`.

# Arguments
- `x::AbstractArray{<:Real}`: The time series data.
- `m::AbstractVector{Bool}`: A boolean vector indicating which data points are flagged as errors.

# Returns
- `x::AbstractArray{<:Real}`: The modified time series `x` with interpolated values for the flagged errors.
"""
function interpolate_errors!(x::AbstractArray{<:Real}, m::AbstractVector{Bool})
    n = length(x)
    t_i(i) = 2 * (i - 1) / (n - 1) - 1
    t = t_i.(1:n)
    m_i = .!m # Invert Mask
    itp = LinearInterpolation(x[m_i], t[m_i]; extrapolation = ExtrapolationType.Linear)
    x[m] .= itp.(t[m])
    return x
end


"""
    optim_timelag(x, y, fc, fs)

Compute the optimal time lag between two signals `x` and `y` using a frequency-domain approach.

# Arguments
- `x::AbstractVector{Float64}`: First input signal.
- `y::AbstractVector{Float64}`: Second input signal.
- `fc::Real`: Cutoff frequency for the high-pass filter. If `fc` is 0, no high-pass filter is applied.
- `fs::Real`: Sampling frequency of the signals.

# Returns
- `tau::Vector{Int64}`: Vector of time lags.
- `out::Vector{Float64}`: Cross-correlation of the signals in the time domain.
- `found_lag::Int64`: Optimal time lag corresponding to the maximum cross-correlation.

# Description
The function computes the cross-correlation between `x` and `y` in the frequency domain, applies a high-pass filter if `fc` is greater than 0, and then transforms the result back to the time domain. The optimal time lag is determined as the lag corresponding to the maximum absolute value of the cross-correlation.

# Notes
The expected time-lag should be lower than half the size of the signals, i.e. `abs(tlag) < div(lengh(x,2)`.

# Examples
```julia
x = randn(100)
y = circshift(x,-10)
tau, out, found_lag = optim_timelag(x, y, 0.1, 20)
found_lag == 10
```
"""
function optim_timelag(
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    fc::Real,
    fs::Real,
)
    length(x) == length(y) || throw(error("Signals must be of the same size."))
    N = length(x)
    L = div(N, 2) + 1 # Analytical fft size
    w = (0:(L-1)) / N
    if fc > 0
        w0 = fc / fs
        low_pass = exp.(-(3 * log(10) / 20) * (w / w0) .^ 2) # -3dB at w=w0
        high_pass = abs2.(1 .- low_pass)
    elseif fc == 0
        high_pass = 1
    end
    tau = 0:(L-1)
    if iseven(N)
        tau = vcat(tau, reverse(-tau .- 1)[3:end])
    else
        tau = vcat(tau, reverse(-tau .- 1)[2:end])
    end
    out = irfft(rfft(x) .* conj(rfft(y)) .* high_pass, N)
    found_lag = tau[argmax(abs.(out))]
    return (tau, out, found_lag)
end

"""
    planar_fit(X[, per_thresh=1.1])

Perform a planar fit on the velocity signals `X=[u v w]` to align the coordinate system.

# Arguments
- `X::AbstractMatrix{<:Real}`: Matrix containing the three wind velocity signals `u`, `v`, and `w` (columns in that order).
- `per_thresh::Real=1.1`: Threshold for velocity variance ratio (log10 scale) to detect degenerate directions. If all directions have similar energy (`|log(ratio)| < |log(per_thresh)`), rotation is ambiguous.

# Returns
- `X_rot::AbstractMatrix{<:Real}`: Rotated velocity signals, aligned with principal directions.
- `P::AbstractMatrix{<:Real}`: The rotation matrix used.
- `theta::Real`: Rotation angle (in radians) to align `[0;0;1]` to the lowest-energy direction.

# Description
The principal directions are found by singular value decomposition of the non-centered correlation matrix `C=X'*X`. The data is rotated to align with the singular vectors of the diagonalisation. It throws an error if the three directions have similar energy, i.e. if the log ratio of the singular values (variances along each directions) are below the absolute log of the threshold `thresh`. For now, it only aligns the original z-coordinate `[0;0;1]` towards the lowest-energy direction found. The rotation avoids 180° flips by ensuring consistent orientation. An additional rotation in the plane normal to the new z-coordinate should be added to perform a full alignment. This can be done by applying the rotation matrix to the singular vectors of the diagonalisation. If we note `u*` the most energetic direction and `u'` the rotation of `[1;0;0]` then the additional rotation should be done in the plane (`u*`,`u'`) to align `u'` towards `u*`.

# Notes
- Make sure that the data is properly filtered, i.e. that the high frequencies corresponding to the turbulent transport are removed. Remove spikes (-9999) if present
- It assumes that `u` carries the most energy and `w` the least.
- It uses axis-angle rotation (not Euler angles) for simplicity.

# Examples
```julia
using LinearAlgebra
X0 = randn(N, 3) .* [10 1 0.1]
pitch=pi/4
Gy, _ = givens(cos(pitch), sin(pitch), 1, 3) # Rotation matrix
X = X0 * Gy'
X_rot, P, theta= planar_fit(X)
```
"""
function planar_fit(X::AbstractArray{<:Real,2}, thresh = 1.1)
    if thresh < 1
        throw(ArgumentError("The threshold (`thresh`) should be > 1 (current: $thresh)"))
    end
    C = X' * X
    E = svd(C)
    S = E.S # Singular values in descending order
    R = E.Vt
    # If two singular values are of the same order then we may have a problem of orientation, it means that two direction carry the same energy and thus it is impossible to decide.
    # we check abs.(log10(sigma_x/sigma_y))
    sigma_diff = abs.(0.5 * log.(circshift(S, 1) ./ circshift(S, 2)))
    thresh = abs(log(thresh))
    k = sigma_diff .< thresh
    if 2 <= count(k) <= 3
        # If the 3 directions have the same energy, finding a rotation is impossible.
        throw(ErrorRotationAmbiguous(thresh))
    elseif k[1] || k[2] # Rotation ambiguous in the YZ and XZ plane meaning that the rotation of the Z axis is ambiguous
        throw(ErrorRotationAmbiguous(thresh))
    end
    # The main assumption here is that u carry the most of the energy and w the least amount of energy
    lambda_low = S[3]
    c_low = R[3, :]
    vref_low = [0; 0; 1]
    # WIP: implement the full rotation by also rotating toward the highest singular value
    # lambda_high = S[1]
    # c_high = R[1, :]
    # vref_high = [1; 0; 0]
    # We check that it is well oriented (avoid a pi rotation)
    c_low = c_low * sign(vref_low' * c_low)
    # Compute the axis normal to the plane scattered by v and c 
    axis = cross(vref_low, c_low)
    axis = axis / norm(axis, 2)
    # Find the angle of rotation beween v and c 
    theta = acos(vref_low' * c_low)
    # Create the corresponding rotation Matrix
    P = AngleAxis(theta, axis...)
    # @assert isapprox(P * vref_low, c_low) # make sure that we get c_low, by rotating vref_low
    @debug "Rotation info" R c_low axis theta P P * R'
    # TODO: add full alignment with axis_high=c_low and theta_high=acos(P*vref_high,c_high)
    # The horizontal rotation will be normal to the new z axis and the angle is between P*vref_high and c_high
    X_rot = X * P
    return (X_rot, P, theta)
end

struct ErrorRotationAmbiguous <: Exception
    msg::String
    function ErrorRotationAmbiguous(thresh::Real)
        msg = """
          Planar fit failed: Velocity components have similar energy along all directions or in the XZ or ZY plane (ratio < $(thresh)).
          Check data preprocessing (filtering) or adjust `thresh` (current: $(thresh))."""
        new(msg)
    end
end
