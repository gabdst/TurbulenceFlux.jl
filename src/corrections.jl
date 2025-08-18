diff_XY(x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, i::Int, j::Int) =
    i == j ? NaN : (y[i] - y[j]) / (x[i] - x[j])

"""
    median_regression(x)

Repeated median regression.
"""
function median_regression(s)
    x = findall(!isnan, s)
    n = length(x)
    s = s[x]
    a = Matrix{Float64}(undef, (n, n))
    beta_vec = Vector{Float64}(undef, n)
    for i = 1:n, j = i:n
        a[i, j] = diff_XY(x, s, i, j)
    end
    a = Symmetric(a) # upper triangular view
    for i = 1:n
        beta_vec[i] = median(filter(!isnan, view(a, :, i)))
    end
    beta_rm = median(beta_vec)
    mu_rm = median(s .- x * beta_rm)
    return mu_rm, beta_rm
end

"""
    MAD(x)

Median Absolute Deviation of `x`.
"""
function mad(x) end


"""
    despiking(x)

Despiking of signal `x`. Returns a filtered signal along with the mask locating the spikes.
"""
function despiking(x) end



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

Perform a planar fit on the velocity signals `X=[u v w]` to align the coordinate system with the principal directions of the data, focusing on the lowest-energy direction (assumed to be `w`).

# Arguments
- `X::AbstractMatrix{<:Real}`: Matrix containing the three wind velocity signals `u`, `v`, and `w` (columns in that order).
- `per_thresh::Real=1.1`: Threshold for velocity variance ratio (log10 scale) to detect degenerate directions. If all directions have similar energy (`|log(ratio)| < |log(per_thresh)`), rotation is ambiguous.

# Returns
- `X_rot::AbstractMatrix{<:Real}`: Rotated velocity signals, aligned with principal directions.
- `P::AbstractMatrix{<:Real}`: The rotation matrix used.
- `theta::Real`: Rotation angle (in radians) to align `[0;0;1]` to the lowest-energy direction.

# Description
The principal directions are found by singular value decomposition of the non-centered correlation matrix `C=X'*X`. The data is rotated to align with the singular vectors of the diagonalisation. It throws an error if the three directions have similar energy, i.e. if the log ratio of the singular values (variances along each directions) are below the absolute log of the threshold `thresh`. For now, it only aligns the original z-coordinate `[0;0;1]` towards the lowest-energy direction found. The rotation avoids 180° flips by ensuring consistent orientation.

# Notes
- Make sure that the data is properly filtered, i.e. that the high frequencies corresponding to the turbulent transport are removed. Removes spikes (-9999) if present
- It assumes that `u` carries the most energy and `w` the least. For full alignment (highest and lowest energy alignment).
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
