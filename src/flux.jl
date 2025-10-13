"""
    FluxEstimationMethod

Abstract type for defining flux estimation methods. The following methods are defined:

- `ReynoldsEstimation`
- `TurbuLaplacian`
- `TurbuThreshold`

Also see `estimate_flux`.
"""
abstract type FluxEstimationMethod end

"""
    ReynoldsEstimation(;tp::TimeParams,tp_aux::TimeParams) <: FluxEstimationMethod

A `FluxEstimationMethod` based on the Reynolds decomposition.

# Keyword Arguments
- `tp::TimeParams`: parameters used to decompose signals into mean and variable parts.
- `tp_aux::TimeParams`: parameters used to estimate auxilliary variables.
- `sensitivity::Bool=true` : flag for computing sensitivity against time and averaging parameter

# Description
To be used with `estimate_flux` in order to perform flux estimation. The Reynolds decomposition is defined via the time decomposition parameters `TimeParams`. The `tp` parameters are used to estimate the fluxes whil the `tp_aux` parameters are used to estimate auxilliary variables such as mean wind and density.

# Computed Variables
As input of `estimate_flux`, it will return a `FluxEstimate{ReynoldsEstimation}` result containing the following variables
 - `WS`: mean wind speed
 - `RHO`: mean density
 - `USTAR`: defined as `SQRT(U'W'^2+V'W'^2)``
 - `U_SIGMA`, `V_SIGMA`, `W_SIGMA`: time-varying standard deviation of `U`,`V`,`W`
 - `H`: Sensible Heat
 - gas fluxes depending on given inputs (see recognized gas `TurbulenceFlux.gas_variables` and `output_variables` for output nomenclature)
 - quality control variables are given for all above variables (ending with `_QC`) (see `QualityControl` and `FluxEstimate`)

See `TimeParams`, `FluxEstimate` and `estimate_flux`.
"""
@kwdef struct ReynoldsEstimation <: FluxEstimationMethod
    tp::TimeParams
    tp_aux::TimeParams
    sensitivity::Bool = true
end

"""
    TurbuThreshold(;tr_tau,dp,tp_aux) <: FluxEstimationMethod

A FluxEstimationMethod based on the thresholding of the Reynolds tensor in time-frequency space.

# Keyword Arguments
- `dp::DecompParams`: time-frequency decomposition parameters
- `tp_aux::TimeParams`: the parameters used to estimate auxilliary variables such as the mean wind speed and the density.
- `tr_tau::Real`: the threshold used to isolate the vertical turbulent transport.
- `sensitivity::Bool=true` : flag for computing sensitivity against time and averaging parameter

# Description
To be used with `estimate_flux` in order to perform flux estimation. Once the signals are decomposed in time-frequency space (TODO)... 

# Computed Variables
As input of `estimate_flux`, it will return a FluxEstimate{ReynoldsEstimation} result containing the following variables
 - `WS`: mean wind speed
 - `RHO`: mean density
 - `USTAR`: defined as `SQRT(U'W'^2+V'W'^2)``
 - `U_SIGMA, V_SIGMA, W_SIGMA`: time-varying standard deviation of `U,V,W`
 - `ETA`: time varying normalized frequency
 - `TAUW_TF`: the vertical amplitude of the Reynold tensor in time-frequency space
 - `TAUW_TF_M`: a mask in time-frequency space localizing the turbulence
 - `TAUW`: the scale integration of `TAUW_TF` according to `TAUW_TF_M`
 - `H_TF`: Sensible Heat in time-frequency space
 - `H`: the scale integration of `H_TF` according to `TAUW_TF_M`
 - gas fluxes in time-frequency space (ending with _TF) depending on given inputs (see recognized gas `TurbulenceFlux.gas_variables` and `output_variables` for output nomenclature)
 - gas fluxes obtained after scale integration according to `TAUW_TF_M`
 - quality control variables are given for all above variables (ending with `_QC`) (see `QualityControl` and `FluxEstimate`)

See `DecompParams`, `TimeParams`, `FluxEstimate`, `turbulence_mask` and `estimate_flux`
"""
@kwdef struct TurbuThreshold <: FluxEstimationMethod
    tr_tau::Real
    dp::DecompParams
    tp_aux::TimeParams
    sensitivity::Bool = true
end

"""
    TurbuLaplacian(;tr_tau,tr_dtau,dp,tp_aux,span=0.25 ) <: FluxEstimationMethod

A FluxEstimationMethod based on the thresholding and laplacian of the Reynolds tensor in time-frequency space.

# Keyword Arguments
- `dp::DecompParams`: time-frequency decomposition parameters
- `tp_aux::TimeParams`: the parameters used to estimate auxilliary variables such as the mean wind speed and the density.
- `tr_tau::Real`: the threshold used to isolate the vertical turbulent transport.
- `tr_dtau::Real`: the threshold used to isolate the zeros in the laplacian of TAUW_TF
- `span::Real=0.25`: parameter used to influence the locally weighted regression algorithm
- `sensitivity::Bool=true` : flag for computing sensitivity against time and averaging parameter

# Description
To be used with `estimate_flux` in order to perform flux estimation. Once the signals are decomposed in time-frequency space (TODO)...

# Computed Variables
As input of `estimate_flux`, it will return a FluxEstimate{ReynoldsEstimation} result containing the following variables
 - `WS`: mean wind speed
 - `RHO`: mean density
 - `USTAR`: defined as `SQRT(U'W'^2+V'W'^2)``
 - `U_SIGMA, V_SIGMA, W_SIGMA`: time-varying standard deviation of `U,V,W`
 - `ETA`: time varying normalized frequency
 - `TAUW_TF`: the vertical amplitude of the Reynold tensor in time-frequency space
 - `DTAUW_TF`: the laplacian of `TAUW_TF`
 - `SG`: a time varying spectral gap
 - `TAUW_TF_MADVEC`: a mask rejecting the large scale advection 
 - `TAUW_TF_M`: a mask in time-frequency space localizing the turbulence
 - `TAUW`: the scale integration of `TAUW_TF` according to `TAUW_TF_M`
 - `H_TF`: Sensible Heat in time-frequency space
 - `H`: the scale integration of `H_TF` according to `TAUW_TF_M`
 - gas fluxes in time-frequency space (ending with `_TF`) depending on given inputs (see recognized gas `TurbulenceFlux.gas_variables` and `output_variables` for output nomenclature)
 - gas fluxes obtained after scale integration according to `TAUW_TF_M`
 - quality control variables are given for all above variables (ending with `_QC`) (see `QualityControl` and `FluxEstimate`)

See `DecompParams`, `TimeParams`, `FluxEstimate`, `turbulence_mask` and `estimate_flux`
"""
@kwdef struct TurbuLaplacian <: FluxEstimationMethod
    tr_tau::Real
    tr_dtau::Real
    dp::DecompParams
    tp_aux::TimeParams
    span::Real = 0.25
    sensitivity::Bool = true
end

"""
    FluxEstimate{T<:FluxEstimationMethod}

Output of `estimate_flux` with the following fields:

- `estimate::NamedTuple`: variables computed by the method `T`
- `qc::QualityControl`: quality control variables on both input and output variables
- `cp::CorrectionParams`: corrections used and updated before the estimation
- `method::T`: the `FluxEstimationMethod` used

"""
@kwdef struct FluxEstimate{T<:FluxEstimationMethod}
    estimate::NamedTuple
    qc::QualityControl
    cp::CorrectionParams
    method::T
end

function tofluxunits(F::AbstractArray, density::AbstractVector, fluxtype::Symbol)
    if fluxtype == :H
        F = F .* density * C_p #W/m2
    elseif fluxtype == :LE
        F = F .* density * LAMBDA #W/m2
    elseif fluxtype == :FC
        F = F .* density # umol/m2/s
    elseif fluxtype == :TAUW
    elseif fluxtype == :TAU
    end
    return F
end

function normalized_frequency(
    frequencies::AbstractVector{<:Real},
    mean_wind::AbstractVector{<:Real},
    distance::Real,
)
    return distance * frequencies' ./ mean_wind
end

"""
    sigmas_wind([u,v,w],tp::TimeParams)

Compute the standard deviations `U_SIGMA, V_SIGMA, W_SIGMA`.
"""
function sigmas_wind(wind_speeds, tp::TimeParams)
    length(wind_speeds) == 3 ||
        throw(error("Three wind speed signals expected, got $(length(wind_speeds))"))
    allequal(length, wind_speeds) ||
        throw(error("Got wind speed signals of different sizes"))
    u, v, w = wind_speeds
    subsampling = false
    up = u - average(u, tp, subsampling)
    vp = v - average(v, tp, subsampling)
    wp = w - average(w, tp, subsampling)
    U_SIGMA = sqrt.(average(up .^ 2, tp))
    V_SIGMA = sqrt.(average(vp .^ 2, tp))
    W_SIGMA = sqrt.(average(wp .^ 2, tp))
    return U_SIGMA, V_SIGMA, W_SIGMA
end

"""
    ustar([u,v,w],tp::TimeParams)

Compute `USTAR`.
"""
function ustar(wind_speeds::AbstractVector{<:AbstractVector{<:Real}}, tp::TimeParams)
    length(wind_speeds) == 3 ||
        throw(error("Three wind speed signals expected, got $(length(wind_speeds))"))
    allequal(length, wind_speeds) ||
        throw(error("Got wind speed signals of different sizes"))
    u, v, w = wind_speeds
    subsampling = false
    up = u - average(u, tp, subsampling)
    vp = v - average(v, tp, subsampling)
    wp = w - average(w, tp, subsampling)
    uw = average(up .* wp, tp)
    vw = average(vp .* wp, tp)
    USTAR = sqrt.(uw .^ 2 .+ vw .^ 2)
    return USTAR
end

"""
    mean_wind([u,v,w],time_params)

Compute the mean wind speed `WS`.
"""
function mean_wind(wind_speeds::AbstractVector{<:AbstractVector{<:Real}}, tp::TimeParams)
    length(wind_speeds) == 3 ||
        throw(error("Three wind speed signals expected, got $(length(wind_speeds))"))
    allequal(length, wind_speeds) ||
        throw(error("Got wind speed signals of different sizes"))
    mw = sqrt.(sum(map(x -> abs2.(average(x, tp)), wind_speeds)))
    return mw
end

"""
    mean_density(PA,TA,tp::TimeParams)

Compute `RHO` given the ambient pressure and temperature `PA` and `TA`.
"""
function mean_density(PA::AbstractArray{<:Real}, TA::AbstractArray{<:Real}, tp::TimeParams)
    length(PA) == length(TA) || throw(error("Pressure and Temperature of different size"))
    return average(PA, tp) ./ (R * average(TA, tp))
end


"""
  flux_scale_integral(scalo,mask)

Integrate along the scales the scalogram `scalo` given `mask`.
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
    flux_scalogram(W,[S1,S2,...],dp::DecompParams)

Compute the cross-scalogram of `WS1`, `WS2`, ... .
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
    return cross_scalogram(L, C, dp, gmw, avg_kernel)
end

_tauw(ww, wu, wv) = sqrt.(ww .^ 2 .+ wu .^ 2 .+ wv .^ 2)

"""
    reynolds_w_scalogram([u,v,w],decomp_params)

Compute the vertical amplitude of the reynolds stress tensor `TAUW`.
"""
function reynolds_w_scalogram(s::AbstractVector{<:AbstractVector{<:Real}}, dp::DecompParams)
    u, v, w = s
    scalos = flux_scalogram(w, [u, v], dp)
    ww = scalos[(1, 1)]
    wu = scalos[(1, 2)]
    wv = scalos[(1, 3)]
    return _tauw(ww, wu, wv)
end

"""
    turbulence_mask(TAUW_TF,method::Union{TurbuLaplacian,TurbuThreshold},mask_error)

Compute the turbulence mask `TAUW_TF_M` given the estimation `method`.
"""
function turbulence_mask(
    tauw::AbstractArray{<:Real,2},
    method::TurbuThreshold,
    mask_error::Union{Nothing,AbstractArray{Bool}} = nothing,
)
    (; tr_tau) = method
    mask = tauw .> tr_tau
    if !isnothing(mask_error)
        mask = mask .&& .!mask_error
    end
    return (; TAUW_TF_M = mask)
end

function turbulence_mask(
    tauw::AbstractArray{<:Real,2},
    WS::AbstractArray{<:Real,1},
    aux::AuxVars,
    method::TurbuLaplacian,
    mask_error::Union{Nothing,AbstractArray{Bool}} = nothing,
)
    (; tr_tau, tr_dtau, span, dp) = method
    (; z_d) = aux
    S = size(tauw)
    freq_peaks = frequency_peaks(dp)
    length(freq_peaks) == S[2] || throw(
        error(
            "The number of scales are wrong in tauw, expected :$(length(freq_peaks)) got: $(S[2])",
        ),
    )
    length(WS) == S[1] || throw(
        error(
            "The size along time dimension is wrong, expected: $(length(mean_wind)) got:$(S[1])",
        ),
    )

    eta = normalized_frequency(freq_peaks, WS, z_d)
    t = LinRange(0, 1, size(tauw, 1))

    ltauw = log10.(tauw)
    ltauw_v = view(ltauw, :) # vec view

    # Reject low-pass filter i.e. at freq_p[end]
    mask = trues(S)
    mask[:, freq_peaks.==0] .= false

    adj_mat = grid_adj_mat(S, mask) # 9-point grid adjacency matrix with removed vertices from mask
    weights_mat = adj_mat # Using the adjacency matrix as the weight matrix amounts to compute a normal laplacian

    g = MyGraph(adj_mat, weights_mat)
    L = laplacian_matrix(g)
    dtau = reshape(L * ltauw_v, S) # the laplacian is zero where mask is false
    τ_mapped = reshape(ltauw_v, S)

    detected = reshape(dtau .> tr_dtau, S) # Look at (t,eta) points with important minimas
    if !isnothing(mask_error)
        detected[mask_error] .= false # remove points with convolution errors
    end
    detected[(eta.>0)] .= false # remove points above eta = 0
    ts = collect(Iterators.flatten((t[view(detected,:,i)] for i in 1:size(detected,2))))

    if count(detected) == 0
        sg = eta[end,:]
    mask_advec =  mask # Get the mask removing the advection + removing the mean value
    else
    #itp = _interpolate_eta(t[detected[:]], eta[detected[:]], λ,d) # Old way: Bspline interpolation + smoothness regularization to get interpolated value at each time t, extrapolate with constant values on the borders
    itp = _locally_weighted_regression(t[detected], eta[detected], span)
    sg = itp.(t) #Time-Varying spectral gap
    mask_advec = (sg .> eta) .&& mask # Get the mask removing the advection + removing the mean value
    end

    mask_lowcoeff = (log10(tr_tau) .> τ_mapped) .&& mask_advec
    return (;
        TAUW_TF_M = mask_lowcoeff,
        SG = sg,
        TAUW_TF_MADVEC = mask_advec,
        DTAUW_TF = dtau,
    )
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

function merge_qc!(estimate, qc)
    for k in intersect(keys(estimate), keys(qc))
        k_qc = Symbol(k, :_QC)
        estimate[k_qc] = qc[k]
    end
    return estimate
end

function estimate_turbuvar(df::Dict, tp::TimeParams, qc::QualityControl)
    turbuvar = Dict()
    mask = error_mask(tp, qc[:U] .|| qc[:V] .|| qc[:W])
    USTAR = ustar([df[:U], df[:V], df[:W]], tp)
    turbuvar[:USTAR] = USTAR
    update_quality_control!(qc, :USTAR, mask)

    U_SIGMA, V_SIGMA, W_SIGMA = sigmas_wind([df[:U], df[:V], df[:W]], tp)
    turbuvar[:U_SIGMA] = U_SIGMA
    update_quality_control!(qc, :U_SIGMA, error_mask(tp, qc[:U]))

    turbuvar[:V_SIGMA] = V_SIGMA
    update_quality_control!(qc, :V_SIGMA, error_mask(tp, qc[:V]))

    turbuvar[:W_SIGMA] = W_SIGMA
    update_quality_control!(qc, :W_SIGMA, error_mask(tp, qc[:W]))
    return turbuvar
end

function estimate_auxvar(df::Dict, tp::TimeParams, qc::QualityControl)
    auxvars = Dict()

    auxvars[:RHO] = mean_density(df[:PA], df[:TA], tp)
    update_quality_control!(qc, :RHO, error_mask(tp, qc[:TA] .|| qc[:PA]))

    mask = error_mask(tp, qc[:U] .|| qc[:V] .|| qc[:W])
    WS = mean_wind([df[:U], df[:V], df[:W]], tp)
    auxvars[:WS] = WS
    update_quality_control!(qc, :WS, mask)
    return auxvars
end


"""
    estimate_flux(;df::Dict,aux::AuxVars,cp::CorrectionParams,method::FluxEstimationMethod)

Estimate fluxes given input and auxilliary variables in `df` and `aux`.

# Keyword Arguments
 - `df::Dict`: a dictionnary mapping keys of type `Symbol` to arrays of data. The naming and unit convention used is given in `mandatory_variables` and `gas_variables`.
 - `aux::AuxVars`:  a struct for specifying auxilliary variables such as the sampling frequency. See `AuxVars`.
 - `cp::CorrectionParams`:

"""
estimate_flux(; df, aux, cp, method) = estimate_flux(df, aux, cp, method)
function estimate_flux(
    df::Dict,
    aux::AuxVars,
    cp::CorrectionParams,
    method::ReynoldsEstimation,
)
    (; tp, tp_aux) = method
    df = deepcopy(df)
    check_variables(df)
    var_names = get_var_names(df)
    work_dim = length(df[:TIMESTAMP])
    time_sampling = 1:tp.dt:work_dim
    estimate = Dict() # Prepare Output
    estimate[:TIMESTAMP] = df[:TIMESTAMP][time_sampling]
    df, qc = apply_correction!(df, cp, aux)

    auxvars = estimate_auxvar(df, tp_aux, qc)
    merge!(estimate, auxvars)
    turbuvar = estimate_turbuvar(df, tp, qc)
    merge!(estimate, turbuvar)

    subsampling = false

    wp = df[:W] - average(df[:W], tp, subsampling)

    tap = df[:TA] - average(df[:TA], tp, subsampling)
    H = average(tap .* wp, tp)
    H = tofluxunits(H, estimate[:RHO], :H)
    update_quality_control!(qc, :H, error_mask(tp, qc[:TA] .|| qc[:W]) .|| qc[:RHO])
    estimate[:H] = H

    if :CO2 in var_names
        g = df[:CO2]
        gp = g - average(g, tp, subsampling)
        FC = average(gp .* wp, tp)
        FC = tofluxunits(FC, estimate[:RHO], :FC)
        update_quality_control!(qc, :FC, error_mask(tp, qc[:CO2] .|| qc[:W]) .|| qc[:RHO])
        estimate[:FC] = FC
    end

    if :H2O in var_names
        g = df[:H2O]
        gp = g - average(g, tp, subsampling)
        LE = average(gp .* wp, tp)
        LE = tofluxunits(LE, estimate[:RHO], :LE)
        update_quality_control!(qc, :LE, error_mask(tp, qc[:H2O] .|| qc[:W]) .|| qc[:RHO])
        estimate[:LE] = LE
    end
    estimate = to_nt(merge_qc!(estimate, qc))
    return FluxEstimate(; estimate, qc, cp, method)
end

function estimate_flux(
    df::Dict,
    aux::AuxVars,
    cp::CorrectionParams,
    method::Union{TurbuThreshold,TurbuLaplacian},
)
    (; dp, tp_aux) = method
    (; tp) = dp
    df = deepcopy(df)
    check_variables(df)
    var_names = get_var_names(df)
    work_dim = length(df[:TIMESTAMP])
    dp.tp.dt == tp_aux.dt || throw(
        error(
            "Different time sampling given: $(dp.tp.dt) against tp_aux.dt = $(tp_aux.dt)",
        ),
    )
    time_sampling = 1:tp_aux.dt:work_dim
    estimate = Dict()
    estimate[:TIMESTAMP] = df[:TIMESTAMP][time_sampling]
    df, qc = apply_correction!(df, cp, aux)
    auxvars = estimate_auxvar(df, tp_aux, qc)
    merge!(estimate, auxvars)
    turbuvar = estimate_turbuvar(df, tp, qc)
    merge!(estimate, turbuvar)

    C = Dict((:W, :W) => :WW_TF, (:U, :W) => :UW_TF, (:V, :W) => :VW_TF, (:W, :TA) => :H_TF)
    if :CO2 in var_names
        C[(:W, :CO2)] = :FC_TF
    end
    if :H2O in var_names
        C[(:W, :H2O)] = :LE_TF
    end
    L = Dict(k => df[k] for k in unique(Iterators.flatten(collect(keys(C)))))
    @info "Cross-Scalogram computation for $(values(C))"
    scalo = cross_scalogram(L, C, dp)
    for c in keys(scalo)
        scalo[c] = tofluxunits(scalo[c], estimate[:RHO], c)
    end
    merge!(estimate, scalo)
    estimate[:TAUW_TF] = _tauw(scalo[:WW_TF], scalo[:UW_TF], scalo[:VW_TF])

    # Quality Control via error propagation
    L_qc = Dict{Symbol,Array{Bool}}()
    C_qc = Dict{Tuple{Symbol,Symbol},Symbol}()
    for c in keys(C)
        x, y = c
        xq = Symbol(x, :_QC)
        yq = Symbol(y, :_QC)
        v = Symbol(C[c], :_QC)
        C_qc[(xq, yq)] = v

        # Add fictitious errors on the border
        qc_x = copy(qc[x])
        qc_x[1] = qc_x[end] = true
        qc_y = copy(qc[y])
        qc_y[1] = qc_y[end] = true
        L_qc[xq] = L_qc[yq] = qc[x] .|| qc[y]
    end
    scalo_qc = cross_scalogram(L_qc, C_qc, dp) #To look at error propagation in time-scale
    for c in keys(scalo_qc)
        qc[c] = sparse(abs.(scalo_qc[c]) .> 1e-6) .|| qc[:RHO]
    end
    qc[:TAUW_TF_QC] = qc[:WW_TF_QC] .|| qc[:UW_TF_QC] .|| qc[:VW_TF_QC]

    if method isa TurbuLaplacian
        tm = turbulence_mask(estimate[:TAUW_TF], estimate[:WS], aux, method, qc[:TAUW_TF_QC])
        estimate[:SG] = tm.SG
        estimate[:DTAUW_TF] = tm.DTAUW_TF
        estimate[:TAUW_TF_M] = tm.TAUW_TF_M
        estimate[:TAUW_TF_MADVEC] = tm.TAUW_TF_MADVEC
    elseif method isa TurbuThreshold
        tm = turbulence_mask(scalo[:TAUW], method)
        estimate[:TAUW_TF_M] = tm.TAUW_TF_M
    end
    if method.sensitivity
        C_dp =
            Dict(k => Symbol(split(string(v), "_TF")[1], "_DSIGMA", "_TF") for (k, v) in C)
        C_dt = Dict(k => Symbol(split(string(v), "_TF")[1], "_DT", "_TF") for (k, v) in C)
        dp_scalo = dp_cross_scalogram(L, C_dp, dp)
        dt_scalo = dt_cross_scalogram(L, C_dt, dp)
        for c in keys(dp_scalo)
            dp_scalo[c] = tofluxunits(dp_scalo[c], estimate[:RHO], c)
        end
        for c in keys(dt_scalo)
            dt_scalo[c] = tofluxunits(dt_scalo[c], estimate[:RHO], c)
        end
        merge!(scalo, dp_scalo, dt_scalo)
    end
    # Scale Integral
    for c in values(C)
        estimate[c] = flux_scale_integral(scalo[c], estimate[:TAUW_TF_M])
        mask = sparse(flux_scale_integral(qc[Symbol(c,:_QC)], estimate[:TAUW_TF_M]))
        var = Symbol(split(string(c), "_TF")[1])
        update_quality_control!(qc, var, mask)
    end
    freq_peaks = frequency_peaks(dp)
    estimate[:ETA] = normalized_frequency(freq_peaks, estimate[:WS], aux.z_d)
    estimate = to_nt(merge_qc!(estimate, qc))
    return FluxEstimate(; estimate, qc, cp, method)
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

# function map_idx(CI, mapping)
#     X = Array{Float64}(undef, length(CI))
#     Y = Array{Float64}(undef, length(CI))
#     Z = Array{Float64}(undef, length(CI))
#     for (i, c) in enumerate(CI)
#         x, y, z = mapping(c)
#         X[i] = x
#         Y[i] = y
#         Z[i] = z
#     end
#     return (X, Y, Z)
# end
