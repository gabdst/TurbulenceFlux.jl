using Statistics
using Unitful: Time
using StatsBase
using PhysicalConstants
using Unitful
import Loess

const LAMBDA = 40660 / 1000 # "J.mmol^-1" latent heat of evaporation of water
const C_p = 29.07 # Molar Heat Capacity at constant pressure J.mol^-1.K^-1
const R = ustrip(PhysicalConstants.CODATA2018.R)

"Convert vector `F` to flux of type `fluxtype` given the density `density`"
function tofluxunits(F, density, fluxtype)
    # Using naming convention of FLUXNET
    if fluxtype == :H
        F = F .* density * C_p
        units = u"J/m^2/s"
    elseif fluxtype == :LE
        F = F .* density * LAMBDA
        units = u"J/m^2/s"
    elseif fluxtype == :FC
        F = F .* density
        units = u"μmol" / u"m^2" / u"s"
    elseif fluxtype == :TAUW # No convention here calling it TAUW instead of TAU
        units = u"m^2" / u"s^2"
    else
        throw(error("wrong flux type"))
    end
    return (F, units)
end

function map_idx(CI, mapping)
    X = Array{Float64}(undef, length(CI))
    Y = Array{Float64}(undef, length(CI))
    Z = Array{Float64}(undef, length(CI))
    for (i, c) in enumerate(CI)
        x, y, z = mapping(c)
        X[i] = x
        Y[i] = y
        Z[i] = z
    end
    return (X, Y, Z)
end

"""
  time_integrate_flux(decomp,mask)

Integrate along the scale domain the flux `decomp` decomposed in the time-scale domain given the integration mask `mask`.

Optionnaly a symbol `fluxtype`can be given to indicate the type of flux such that it is converted to common flux units, see `tofluxunits`.
"""
function time_integrate_flux(decomp, mask)
    size(mask) == size(decomp) || throw(error("Wrong size between mask and decomp"))
    F = Vector{Float64}(undef, size(decomp, 1))
    for j in axes(F, 1)
        values = decomp[j, mask[j, :]]
        if isempty(values)
            F[j] = NaN
        else
            F[j] = sum(decomp[j, mask[j, :]])
        end
    end
    return F
end


"""
    compute_wind_amplitude(wind_speeds,time_params)

Compute the wind amplitude given the three wind speed components stored in `wind_speeds`, with the averaging convolutional kernel parameters `time_params`.
"""
function compute_wind_amplitude(wind_speeds::AbstractArray{<:Real,2}, tp::TimeParams)
    input_dim = size(wind_speeds)
    work_dim = input_dim[1]
    avg_kernel = averaging_kernel(tp)
    (; dt, kernel_dim) = tp
    KernelC = CConv(Float64, input_dim, kernel_dim)
    time_sampling = 1:dt:work_dim
    wind_amplitude = KernelC(wind_speeds, avg_kernel)[time_sampling, 1, :]
    wind_amplitude = mapslices(x -> sqrt(sum(abs2, x)), wind_amplitude, dims = 2)
    wind_amplitude = dropdims(wind_amplitude, dims = 2)
    return wind_amplitude
end

"""
    compute_density(P,T,time_params)

Compute the density given the pressure `P` (Pa) and the temperature `T` (K) with the averaging convolutional kernel parameters `time_params`.
"""
function compute_density(P::AbstractArray{<:Real}, T::AbstractArray{<:Real}, tp::TimeParams)
    length(P) == length(T) || throw(error("Signals of different size"))
    work_dim = length(P)
    input_dim = (work_dim, 1)
    avg_kernel = averaging_kernel(tp)
    (; dt, kernel_dim) = tp
    KernelC = CConv(Float64, input_dim, kernel_dim)
    time_sampling = 1:dt:work_dim
    density = P ./ (R * T)
    density = KernelC(density, avg_kernel)[time_sampling]
    return density
end


"""
    timescale_flux_decomp(w,s,time_params,scale_params;with_info=false)

Compute the time-scale decomposition of the flux `ws` given averaging and wavelet kernel parameters `time_params` and `scale_params`.

# Arguments

  - `w::Vector`: velocity signal (e.g. the vertical wind speed)
  - `s::Vector{Vector}`: scalar signals (temperature, gaz concentration,...)
  - `time_params::NamedTuple`: TODO 
  - `scale_params::NamedTuple`: TODO 
  - `with_info::Bool=false`: Output informations about the decomposition
  - `func=nothing`: Apply function `func` before averaging step
"""
function timescale_flux_decomp(
    w::AbstractArray{<:Real},
    s::Vector{<:AbstractArray{<:Real}},
    tp::TimeParams,
    sp::ScaleParams;
    with_info = false,
)
    allequal(length, s) && length(w) == length(s[1]) ||
        throw(error("Signals must be of the same size."))
    avg_kernel = averaging_kernel(tp)
    gmw_frame = GMWFrame(sp)
    C = [(1, i + 1) for i = 1:length(s)]
    L = [w, s...]

    out = cross_scalogram(L, C, tp.dt, gmw_frame, avg_kernel)
    if with_info
        return (; info = (gmw_frame, avg_kernel), out = out)
    else
        return out
    end

end

"""
  get_timescale_mask(work_dim,sigma_waves,sigma_averaging,factor=(3,3),max_sigma=false)

Construct a time-scale mask given the time deviation of the wavelet filters and the averaging kernel.

# Arguments

 - `work_dim`: dimension of analysis.
 - `sigma_waves`: wavelet filters time-deviation
 - `sigma_averaging`: averaging kernel time-deviation
 - `factor=(3,3)`: amounts by which time-deviations of wavelets and filters are multiplied
 - `max_sigma=false`: return a border error mask with the maximum time deviation
"""
function get_timescale_mask(
    work_dim,
    sigma_waves,
    sigma_averaging,
    factor = (3, 3),
    max_sigma = false,
)
    mask = falses(work_dim, length(sigma_waves))
    max_sigma_val =
        ceil(Int, maximum(sigma_waves) * factor[1] + sigma_averaging * factor[2])
    for (i, s) in enumerate(sigma_waves)
        if max_sigma
            s = max_sigma_val
        else
            s = ceil(Int, s * factor[1] + sigma_averaging * factor[2])
        end
        mask[1:s, i] .= true
        mask[(end-s+1):end, i] .= true
    end
    return mask
end


"""
    amplitude_reynolds_w(u,v,w,time_params,scale_params)

Compute the amplitude of the vertical components of the Reynold's tensor using the three wind speed components `u`,`v` and `w` and the time-scale decomposition parameters `time_params` and `scale_params`.

# Arguments

  - `u,v,w::Vector`: wind speed components signal (e.g. the vertical wind speed)
  - `time_params::NamedTuple`: Named Tuple of the parameters for initializing the averaging convolutional kernel, see`init_averaging_conv_kernel`
  - `scale_params::NamedTuple`: Named Tuple of parameters for initializing the wavelet convolutional kernel, see `init_wave_conv_kernel`
"""
function amplitude_reynolds_w(
    u::AbstractArray{<:Real},
    v::AbstractArray{<:Real},
    w::AbstractArray{<:Real},
    tp::TimeParams,
    sp::ScaleParams;
    with_info = false,
)
    return timescale_flux_decomp(w, [w, u, v], tp, sp; with_info)
end

"""
    turbulence_mask(u,v,w,time_params,scale_params,method,method_params...)

Estimate a time-scale mask of the vertical turbulent transport using the three wind speed components `u`,`v` and `w` using the time-scale decomposition parameters `time_params` and `scale_params` (see `init_averaging_conv_kernel` and `init_wave_conv_kernel`) and the turbulence extraction methods `method` with parameters `method_params`.

# Arguments

  - `u,v,w::Vector`: wind speed components signal (e.g. the vertical wind speed)
  - `time_params::NamedTuple`: Named Tuple of the parameters for initializing the averaging convolutional kernel, see`init_averaging_conv_kernel`
  - `scale_params::NamedTuple`: Named Tuple of parameters for initializing the wavelet convolutional kernel`, see `init_wave_conv_kernel`
  - `method::Function`: Method used to extract the turbulent transport signal, see `turbu_extract_threshold`,`turbu_extract_laplacian` and `turbu_extract_diffusion`.
"""
function turbulence_mask(
    u::AbstractArray{<:Real},
    v::AbstractArray{<:Real},
    w::AbstractArray{<:Real},
    tp::TimeParams,
    sp::ScaleParams,
    method::Function,
    method_params...,
)
    (; info, tauw) = amplitude_reynolds_w(u, v, w, tp, sp; with_info = true)
    return method(tauw; method_params...)
end
function turbulence_mask(tauw, ; method::Function, method_params...)
    return method(tauw; method_params...)
end

function turbu_extract_threshold(tauw; threshold)
    mask = tauw .> threshold
    return mask
end

function _locally_weighted_regression(t, eta, span = 0.25)
    model = Loess.loess(t, eta, span = span)
    tmin, tmax = extrema(t)
    function g(t)
        if t < tmin
            return Loess.predict(model, tmin)
        elseif t > tmax
            return Loess.predict(model, tmax)
        else
            return Loess.predict(model, t)
        end
    end
    return g
end

function turbu_extract_laplacian(
    t,
    eta,
    log10tauw;
    tr_dtau = 1,
    tr_tau = 1e-3,
    span = 0.25,
    mask_error = falses(size(log10tauw)),
)

    S = size(log10tauw)
    τ_mapped = view(log10tauw, :)

    # Reject low-pass filter i.e. at freq_p[end]
    mask = trues(S)
    mask[:, end] .= false

    adj_mat = grid_adj_mat(S, mask) # 9-point grid adjacency matrix with removed vertices from mask
    weights_mat = adj_mat # Using the adjacency matrix as the weight matrix amounts to compute a normal laplacian

    g = MyGraph(adj_mat, weights_mat)
    L = laplacian_matrix(g)
    dtau = reshape(L * τ_mapped, S) # the laplacian is zero where mask is false
    τ_mapped = reshape(τ_mapped, S)

    detected = reshape(tr_dtau .< dtau, S) # Look at (t,eta) points with important minimas
    detected[mask_error] .= false # remove points with convolution errors
    detected[(eta.>0)] .= false # remove points above eta = 0

    #itp = _interpolate_eta(t[detected[:]], eta[detected[:]], λ,d) # Old way: Bspline interpolation + smoothness regularization to get interpolated value at each time t, extrapolate with constant values on the borders
    itp = _locally_weighted_regression(t[detected], eta[detected], span)

    mask_advec = (itp.(t) .< eta) .&& mask # Get the mask removing the advection + removing the mean value
    mask_lowcoeff = (log10(tr_tau) .< τ_mapped) .&& mask_advec
    masks = (; minimas = detected, advection = mask_advec, turbulence = mask_lowcoeff)

    return (masks, dtau, itp)
end

"""
   WIP 
"""
function turbu_extract_diffusion(
    tauw;
    time_sampling,
    freq_peak,
    ref_dist = 1,
    mean_wind = nothing,
)
    S = size(tauw)
    CI = CartesianIndices(S)
    # Reject low-pass filter i.e. at freq_p[end]
    mask = trues(S)
    mask[:, end] .= false
    if isnothing(mean_wind)
        vertex_mapping =
            (c::CartesianIndex) -> Float64[
                time_sampling[c[1]],
                log.(freq_peak[c[2]] * ref_dist),
                log(tauw[c[1], c[2]]),
            ]
    else
        vertex_mapping =
            (c::CartesianIndex) -> Float64[
                time_sampling[c[1]],
                log.(freq_peak[c[2]] * ref_dist / mean_wind[c[1]]),
                log(tauw[c[1], c[2]]),
            ]
    end
    t, eta, tau_mapped = map_idx(CI, vertex_mapping)
    adj_mat = grid_adj_mat(S, mask)
    sigma_tau = std(tau_mapped[mask[:]])
    function weight_func(i::Int, j::Int)
        c_i = CI[i]
        c_j = CI[j]
        v_i = vertex_mapping(c_i)[3]
        v_j = vertex_mapping(c_j)[3]
        v = exp(-(v_i - v_j) / sigma_tau) # Asymetric Potential
        return v
    end
    weights_mat = generate_weight_mat(adj_mat, weight_func; normalize = true)
    g = MyGraph(adj_mat, weights_mat)
    tau_rey = (reshape(t, S), reshape(eta, S), reshape(tau_mapped, S))
    return (tau_rey, g)
    #
    #  s=-0.1 .< Y .< 0.1
    #  s=sparse(vec(s))
    #  M=sum(s)
    #  func_acc(s,i)=begin
    #      x=s .+ droptol!(g.weights*s,1e-6)
    #      x=x*(M/sum(x))
    #      return x
    #  end
    #  #all_s=accumulate(func_acc,1:10,init=s);
    #  @warn  println("Not Fully implemented yet")
    #  return (g,Δv,sigma,(X,Y,tau_mapped))
end


"""
    flux_estimation(data,z_d,fs,time_params,scale_params)

Wavelet based estimation of the flux given averaging parameters `time_params` and wavelet parameters `scale_params`.

# Arguments
 - `data::DataFrame`: wind speed, pressure and gas concentrations measurements
 - `z_d::Real`: measurement height above the zeros place displacement height
 - `fs::Integer`: sampling frequency
 - `time_params::NamedTuple`: averaging parameters
 - `scale_params::NamedTuple`: wavelet decomposition parameters
"""
function flux_estimation(
    data;
    z_d,
    fs,
    time_params,
    scale_params,
    time_params_mean_wind = time_params,
    time_params_density = time_params,
    time_params_turbu = time_params,
    freq_tl = (0.1, 1),
    dates = (nothing, nothing),
    with_decomp = false,
    analysis_range = Colon(),
    kwargs...,
)
    sdate, edate = dates
    t0 = time()
    (; U, V, W, T, CO2, H2O, P) = data
    work_dim = length(U)

    contains_nan = any(i -> !isnothing(findfirst(isnan, data[:, i])), 2:size(data, 2))
    if contains_nan
        throw(error("Data contains NaN values. Aborting."))
    end
    T = T .+ 274.15 # °C TO K
    P = 1000 * P # kPa to Pa

    time_h = (0:(work_dim-1))
    time_h = time_h ./ (60 * 60 * fs)
    wind_speeds = hcat(U, V)

    # Wind Amplitude signal and density signal
    mean_wind = compute_wind_amplitude(wind_speeds, time_params_mean_wind)
    density = compute_density(P, T, time_params_density)
    # Time-Lag optimisation
    τ_max = 60 * fs # 1min max timelag search
    τ_arr, corr_H2O = optim_timelag(W, H2O, scale_params, freq_tl, τ_max)
    _, corr_CO2 = optim_timelag(W, CO2, scale_params, freq_tl, τ_max)

    tl_max = -2 * fs # 2s maximum timelag 
    m_tl = τ_arr .<= 0 # search only maximum in negative timelag, gas analyser is always late
    tl_H2O = τ_arr[m_tl][argmax(abs.(corr_H2O[m_tl]))]
    tl_H2O = tl_H2O < tl_max ? 0 : tl_H2O # If we reach a maximum, better not to take it
    tl_CO2 = τ_arr[m_tl][argmax(abs.(corr_CO2[m_tl]))]
    tl_CO2 = tl_CO2 < tl_max ? 0 : tl_CO2
    timelags = Dict(:H2O => (tl_H2O, τ_arr, corr_H2O), :CO2 => (tl_CO2, τ_arr, corr_CO2))
    circshift!(H2O, timelags[:H2O][1])
    circshift!(CO2, timelags[:CO2][1])

    max_tl = max(abs(timelags[:H2O][1]), abs(timelags[:CO2][1]))

    # Time-Scale Analyses
    time_sampling, (freq_peak, sigma_t), decomp_FC =
        timescale_flux_decomp(W, CO2, time_params, scale_params; with_info = true)
    _, _, decomp_H = timescale_flux_decomp(W, T, time_params, scale_params)
    _, _, decomp_LE = timescale_flux_decomp(W, H2O, time_params, scale_params)
    _, _, decomp_TAUW = amplitude_reynolds_w(U, V, W, time_params_turbu, scale_params)
    Z = log10.(decomp_TAUW)

    sigma_t = (sigma_t[1] .+ max_tl, sigma_t[2]) # add maximum timelag estimated as border error 
    mask_sigma_t = get_timescale_mask(work_dim, sigma_t..., (4, 4), false)[time_sampling, :] # convolution border errors mask

    to_eta(i_t, j_ξ) = log10(((z_d * freq_peak[j_ξ]) / mean_wind[i_t]))
    S = size(decomp_H) # Dimension
    CI = CartesianIndices(S)

    t = map(c -> time_h[time_sampling[c[1]]], CI) # get the time values
    eta = map(c -> to_eta(c[1], c[2]), CI) # get the normalized freq

    (masks, Δτ, itp) =
        turbu_extract_laplacian(t, eta, Z, δ_Δτ = 1, δ_τ = 1e-3, mask_error = mask_sigma_t)
    mask_minima, mask_NOADVEC, mask_TURBU = masks
    advec_line = itp.(time_h[time_sampling])
    advec_line = advec_line[analysis_range]

    mask_analysis = falses(size(decomp_H)) # Restriction to period of analysis
    mask_analysis[analysis_range, 1:(end-1)] .= true # We take everything during the analysis range period and without the first frequency peak at 0
    mask_NOMEAN = copy(mask_analysis) .&& .!(mask_sigma_t) # Remove border errros
    mask_TURBU = mask_TURBU .&& .!(mask_sigma_t) # Remove border errros

    decomp_H, units_H = tofluxunits(decomp_H, density, :H) # convert to conventional units (here W/m2)
    decomp_FC, units_FC = tofluxunits(decomp_FC, density, :FC)
    decomp_LE, units_LE = tofluxunits(decomp_LE, density, :LE)

    H_NOMEAN = time_integrate_flux(decomp_H, mask_NOMEAN)
    LE_NOMEAN = time_integrate_flux(decomp_LE, mask_NOMEAN)
    FC_NOMEAN = time_integrate_flux(decomp_FC, mask_NOMEAN)

    H_NOADVEC = time_integrate_flux(decomp_H, mask_NOADVEC)
    LE_NOADVEC = time_integrate_flux(decomp_LE, mask_NOADVEC)
    FC_NOADVEC = time_integrate_flux(decomp_FC, mask_NOADVEC)

    H_TURBU = time_integrate_flux(decomp_H, mask_TURBU)
    LE_TURBU = time_integrate_flux(decomp_LE, mask_TURBU)
    FC_TURBU = time_integrate_flux(decomp_FC, mask_TURBU)

    if with_decomp
        decomp = Dict(
            pairs((;
                t,
                eta, #in log10
                H = decomp_H,
                FC = decomp_FC,
                LE = decomp_LE,
                TAUW = Z, #in log10
                DELTA_TAUW = Δτ,
                mask_analysis,
                mask_minima,
                mask_NOADVEC,
                mask_TURBU,
                mask_sigma_t,
            )),
        )
    else
        decomp = nothing
    end

    fluxes = Dict(
        pairs((;
            H_NOMEAN = H_NOMEAN[analysis_range],
            LE_NOMEAN = LE_NOMEAN[analysis_range],
            FC_NOMEAN = FC_NOMEAN[analysis_range],
            H_NOADVEC = H_NOADVEC[analysis_range],
            LE_NOADVEC = LE_NOADVEC[analysis_range],
            FC_NOADVEC = FC_NOADVEC[analysis_range],
            H_TURBU = H_TURBU[analysis_range],
            LE_TURBU = LE_TURBU[analysis_range],
            FC_TURBU = FC_TURBU[analysis_range],
        )),
    )

    time_analysis_h = time_h[time_sampling][analysis_range]

    t1 = time()
    res_p = pairs((;
        dates = (sdate, edate),
        time_analysis_h,
        time_execution = t1 - t0,
        advec_line,
        fluxes,
        decomp,
        timelags,
    ))
    results = Dict(res_p)
    return results
end

function optim_timelag(w, s, fc, fs, lag_max)
    #TODO: implement it with an iir filter
    length(w) == length(s) || throw(error("Signals must be of the same size."))
    N = length(w)

    lp = lowpass(fc, fs)
    w_l = applyfilter(lp, w)
    w_s = applyfilter(lp, s)
    w_h = w - w_l

    w_h

    L = div(N, 2) + 1 # Analytical fft size
    w = (0:(L-1)) / N
    w0 = freq_cut
    low_pass = exp.(-(3 * log(10) / 20) * (w / w0) .^ 2) # -3dB at w=w0
    gmw_frame = [gmw_frame..., low_pass]
    freq_peaks = vcat(freq_peaks, 0.0)

    τ = 0:(lag_max-1)
    τ_arr = vcat(reverse(-τ .- 1), τ)

    out = irfft(sum(rfft(w_ξ, 1) .* conj(rfft(θ_ξ, 1)), dims = 2)[:], work_dim)



    out = vcat(out[end-length(τ)+1:end], out[1:length(τ)])
    return (τ_arr, out)
end

## UTILS, TODO: put in utils.jl

function find_nan_regions(F)
    s = Int64[]
    e = Int64[]
    i = 1
    L = length(F)
    while !isnothing(i)
        i = findnext(isnan, F, i)
        if isnothing(i)
            break
        else
            si = i > 1 ? i - 1 : i
            push!(s, si)
        end

        j = findnext(!isnan, F, i)
        sj = isnothing(j) ? L : j
        push!(e, sj)
        i = j # End loop if j==nothing
    end
    return (s, e)
end
