mutable struct TurbulentFluxEstimate
    work_dim::Int64
    mean_wind::Vector{Float64}

end


"""
    mean_wind_speed([u,v,w],time_params)
"""
function mean_wind_speed(
    wind_speeds::AbstractVector{<:AbstractVector{<:Real}},
    tp::TimeParams,
)
    length(wind_speeds) == 3 ||
        throw(error("Three wind speed signals expected, got $(length(wind_speeds))"))
    allequal(length, wind_speeds) ||
        throw(error("Got wind speed signals of different sizes"))
    mean_speed = sqrt.(sum(map(x -> abs2.(average(x, tp)), wind_speeds)))
    return mean_speed
end

"""
    mean_density(P,T,time_params)
"""
function mean_density(P::AbstractArray{<:Real}, T::AbstractArray{<:Real}, tp::TimeParams)
    length(P) == length(T) || throw(error("Signals of different size"))
    density = P ./ (R * T)
    mean_density = average(density, tp)
    return mean_density
end

"Convert vector `F` to flux of type `fluxtype` given the density `density`"
function tofluxunits(F, density, fluxtype)
    # Using naming convention of FLUXNET
    if fluxtype == :H
        F = F .* density * C_p
    elseif fluxtype == :LE
        F = F .* density * LAMBDA
    elseif fluxtype == :FC
        F = F .* density
    elseif fluxtype == :TAUW # No convention here calling it TAUW instead of TAU
    else
        throw(error("wrong flux type"))
    end
    return F
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
  flux_scale_integral(scalo,mask)
"""
function flux_scale_integral(scalo::AbstractArray{<:Real,2}, mask::AbstractArray{Bool,2})
    size(mask) == size(scalo) || throw(error("Wrong size between mask and scalogram"))
    F = Vector{Float64}(undef, size(scalo, 1))
    for j in axes(F, 1)
        values = scalo[j, mask[j, :]]
        if isempty(values)
            F[j] = NaN
        else
            F[j] = sum(values)
        end
    end
    return F
end

"""
    flux_scalogram(w,s,time_params,scale_params;with_info=false)
"""
function flux_scalogram(
    w::AbstractVector{<:Real},
    s::AbstractVector{<:AbstractVector{<:Real}},
    dp::DecompParams,
)
    allequal(length, s) && length(w) == length(s[1]) ||
        throw(error("Signals must be of the same size."))
    (; tp, sp) = dp
    avg_kernel = averaging_kernel(tp)
    gmw = GMWFrame(sp)
    C = [(1, i + 1) for i = 1:length(s)]
    L = [w, s...]
    scalograms = cross_scalogram(L, C, dp, gmw, avg_kernel)
end

"""
    reynolds_w_scalogram([u,v,w],time_params,scale_params)
"""
function reynolds_w_scalogram(s::AbstractVector{<:AbstractVector{<:Real}}, dp::DecompParams)
    u, v, w = s
    scalos = flux_scalogram(w, [u, v], dp)
    ww = scalos[(1, 1)]
    wu = scalos[(1, 2)]
    wv = scalos[(1, 3)]
    return sqrt.(ww .^ 2 .+ wu .^ 2 .+ wv .^ 2)
end

"""
    turbulence_mask([u,v,w],time_params,scale_params,method,method_params...)
"""
function turbulence_mask(
    s::AbstractVector{<:AbstractVector{<:Real}},
    dp::DecompParams,
    method::Function,
    method_params...,
)
    tauw = reynolds_w_scalogram(s, dp)
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

# function turbu_extract_diffusion(
#     tauw;
#     time_sampling,
#     freq_peak,
#     ref_dist = 1,
#     mean_wind = nothing,
# )
#     S = size(tauw)
#     CI = CartesianIndices(S)
#     # Reject low-pass filter i.e. at freq_p[end]
#     mask = trues(S)
#     mask[:, end] .= false
#     if isnothing(mean_wind)
#         vertex_mapping =
#             (c::CartesianIndex) -> Float64[
#                 time_sampling[c[1]],
#                 log.(freq_peak[c[2]] * ref_dist),
#                 log(tauw[c[1], c[2]]),
#             ]
#     else
#         vertex_mapping =
#             (c::CartesianIndex) -> Float64[
#                 time_sampling[c[1]],
#                 log.(freq_peak[c[2]] * ref_dist / mean_wind[c[1]]),
#                 log(tauw[c[1], c[2]]),
#             ]
#     end
#     t, eta, tau_mapped = map_idx(CI, vertex_mapping)
#     adj_mat = grid_adj_mat(S, mask)
#     sigma_tau = std(tau_mapped[mask[:]])
#     function weight_func(i::Int, j::Int)
#         c_i = CI[i]
#         c_j = CI[j]
#         v_i = vertex_mapping(c_i)[3]
#         v_j = vertex_mapping(c_j)[3]
#         v = exp(-(v_i - v_j) / sigma_tau) # Asymetric Potential
#         return v
#     end
#     weights_mat = generate_weight_mat(adj_mat, weight_func; normalize = true)
#     g = MyGraph(adj_mat, weights_mat)
#     tau_rey = (reshape(t, S), reshape(eta, S), reshape(tau_mapped, S))
#     return (tau_rey, g)
#     #
#     #  s=-0.1 .< Y .< 0.1
#     #  s=sparse(vec(s))
#     #  M=sum(s)
#     #  func_acc(s,i)=begin
#     #      x=s .+ droptol!(g.weights*s,1e-6)
#     #      x=x*(M/sum(x))
#     #      return x
#     #  end
#     #  #all_s=accumulate(func_acc,1:10,init=s);
#     #  @warn  println("Not Fully implemented yet")
#     #  return (g,Δv,sigma,(X,Y,tau_mapped))
# end


"""
    estimate_flux(data,z_d,fs,time_params,scale_params)
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
