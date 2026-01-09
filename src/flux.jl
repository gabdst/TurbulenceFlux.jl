"""
    FluxEstimationMethod

Abstract type for defining flux estimation methods. The following methods are defined:

- `ReynoldsEstimation`
- `TurbuLaplacian`
- `TurbuThreshold`

Also see [`estimate_flux`](@ref), and each particular method (e.g. [`ReynoldsEstimation`](@ref)).
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

# Returns
As input of `estimate_flux`, it will return a `FluxEstimate{ReynoldsEstimation}` object containing the following variables
 - `WS`: mean wind speed
 - `RHO`: mean density
 - `USTAR`: defined as `SQRT(U'W'^2+V'W'^2)``
 - `U_SIGMA`, `V_SIGMA`, `W_SIGMA`: time-varying standard deviation of `U`,`V`,`W`
 - `H`: Sensible Heat
 - gas fluxes depending on given inputs (see recognized gas `TurbulenceFlux.gas_variables` and `output_variables` for output nomenclature)
 - quality control variables are given for all above variables (ending with `_QC`) (see `QualityControl` and `FluxEstimate`)

See `TimeParams`, `FluxEstimate` and `estimate_flux`.
"""

"""
    ReynoldsEstimation(; tp::TimeParams, tp_aux::TimeParams) <: FluxEstimationMethod

A `FluxEstimationMethod` using the **Reynolds decomposition** for flux estimation.

# Keyword Arguments
- `tp::TimeParams`: Parameters for decomposing signals into mean and variable components.
- `tp_aux::TimeParams`: Parameters for estimating auxiliary variables (e.g., mean wind, density).

# Description
This method is designed for use with [`estimate_flux`](@ref) to perform flux estimation. The Reynolds decomposition is applied using the time parameters specified in `tp` for flux estimation and `tp_aux` for auxiliary variables.

# Returns
When passed to `estimate_flux`, it returns a `FluxEstimate{ReynoldsEstimation}` object containing:
- `WS`: Mean wind speed
- `RHO`: Mean density
- `USTAR`: Friction velocity, computed as `sqrt(U'W'^2 + V'W'^2)`
- `U_SIGMA`, `V_SIGMA`, `W_SIGMA`: Time-varying standard deviations of `U`, `V`, and `W`
- `H`: Sensible heat flux
- Gas fluxes (depending on input; see [`TurbulenceFlux.gas_variables`](@ref) and [`output_variables`](@ref) for nomenclature)
- Quality control variables for all above (suffixed with `_QC`; see [`QualityControl`](@ref) and [`FluxEstimate`](@ref)).

# See Also
[`TimeParams`](@ref), [`FluxEstimate`](@ref), [`estimate_flux`](@ref)
"""
@kwdef struct ReynoldsEstimation <: FluxEstimationMethod
    tp::TimeParams
    tp_aux::TimeParams
end

"""
    TurbuThreshold(; tr_tau, dp, tp_aux) <: FluxEstimationMethod

A `FluxEstimationMethod` that estimates turbulent fluxes by thresholding the Reynolds tensor in time-frequency space.

# Arguments
- `dp::DecompParams`: Parameters for time-frequency decomposition.
- `tp_aux::TimeParams`: Parameters for estimating auxiliary variables (e.g., mean wind speed, density).
- `tr_tau::Real`: Threshold value for isolating vertical turbulent transport.
- `tp_tau::DecompParams=dp.tp`: Alternative TimeParams for `TAUW_TF`
- `sensitivity::Bool=true`: If `true`, computes sensitivity to time and averaging parameters.

# Output
When passed to `estimate_flux`, this method returns a `FluxEstimate{ReynoldsEstimation}` containing:
- `WS`: Mean wind speed.
- `RHO`: Mean density.
- `USTAR`: Friction velocity, computed as `sqrt(U'W'^2 + V'W'^2)`.
- `U_SIGMA`, `V_SIGMA`, `W_SIGMA`: Time-varying standard deviations of `U`, `V`, and `W`.
- `ETA`: Time-varying normalized frequency.
- `TAUW_TF`: Vertical amplitude of the Reynolds tensor in time-frequency space.
- `TAUW_TF_M`: Mask localizing turbulence in time-frequency space.
- `TAUW`: Scale-integrated `TAUW_TF` according to `TAUW_TF_M`.
- `H_TF`: Sensible heat flux in time-frequency space.
- `H`: Scale-integrated `H_TF` according to `TAUW_TF_M`.
- Time-frequency gas fluxes (suffix `_TF`) for recognized gases (see [`gas_variables`](@ref)).
- Scale-integrated gas fluxes according to `TAUW_TF_M`.
- Quality control variables (suffix `_QC`) for all output variables (see [`QualityControl`](@ref) and [`FluxEstimate`](@ref)).

# See Also
- [`DecompParams`](@ref), [`TimeParams`](@ref), [`FluxEstimate`](@ref), [`turbulence_mask`](@ref), [`estimate_flux`](@ref)
"""
@kwdef struct TurbuThreshold <: FluxEstimationMethod
    tr_tau::Real
    dp::DecompParams
    tp_aux::TimeParams
    tp_tau::TimeParams = dp.tp
    sensitivity::Bool = true
end

"""

    TurbuLaplacian(; tr_tau, tr_dtau, dp, tp_aux, span=0.25) <: FluxEstimationMethod

A `FluxEstimationMethod` that estimates turbulent fluxes by applying thresholding and a Laplacian operator to the Reynolds tensor in time-frequency space.

# Arguments
- `dp::DecompParams`: Parameters for time-frequency decomposition.
- `tp_aux::TimeParams`: Parameters for estimating auxiliary variables (e.g., mean wind speed, density).
- `tr_tau::Real`: Threshold for isolating vertical turbulent transport.
- `tr_dtau::Real`: Threshold for isolating zeros in the Laplacian of `TAUW_TF`.
- `tp_tau::DecompParams=dp.tp`: Alternative TimeParams for `TAUW_TF`
- `span::Real=0.25`: Parameter influencing the locally weighted regression algorithm.
- `sensitivity::Bool=true`: If `true`, computes sensitivity to time and averaging parameters.

# Output
When passed to `estimate_flux`, this method returns a `FluxEstimate{ReynoldsEstimation}` containing:

## Core Variables
- `WS`: Mean wind speed.
- `RHO`: Mean density.
- `USTAR`: Friction velocity, computed as `sqrt(U'W'^2 + V'W'^2)`.
- `U_SIGMA`, `V_SIGMA`, `W_SIGMA`: Time-varying standard deviations of `U`, `V`, and `W`.
- `ETA`: Time-varying normalized frequency.
- `TAUW_TF`: Vertical amplitude of the Reynolds tensor in time-frequency space.
- `DTAUW_TF`: Laplacian of `TAUW_TF`.
- `SG`: Time-varying spectral gap.
- `TAUW_TF_MADVEC`: Mask rejecting large-scale advection.
- `TAUW_TF_M`: Mask localizing turbulence in time-frequency space.
- `TAUW`: Scale-integrated `TAUW_TF` according to `TAUW_TF_M`.
- `H_TF`: Sensible heat flux in time-frequency space.
- `H`: Scale-integrated `H_TF` according to `TAUW_TF_M`.
- Time-frequency gas fluxes (suffix `_TF`) for recognized gases (see [`gas_variables`](@ref)).
- Scale-integrated gas fluxes according to `TAUW_TF_M`.
- Quality control variables (suffix `_QC`) for all output variables (see [`QualityControl`](@ref) and [`FluxEstimate`](@ref)).

# See Also
- [`DecompParams`](@ref), [`TimeParams`](@ref), [`FluxEstimate`](@ref), [`turbulence_mask`](@ref), [`estimate_flux`](@ref)
"""
@kwdef struct TurbuLaplacian <: FluxEstimationMethod
    tr_tau::Real
    tr_dtau::Real
    dp::DecompParams
    tp_aux::TimeParams
    tp_tau::TimeParams = dp.tp
    span::Real = 0.25
    sensitivity::Bool = true
end

@kwdef struct TurbuMorpho <: FluxEstimationMethod
    tr_tau::Real
    a::Real = 1
    N::Integer = 7
    dp::DecompParams
    tp_aux::TimeParams
    tp_tau::TimeParams = dp.tp
    sensitivity::Bool = true
end

"""
    FluxEstimate{T<:FluxEstimationMethod}

Output structure returned by `estimate_flux`, containing the following fields:

# Fields
- `estimate::NamedTuple`: Variables computed by the method `T`.
- `qc::QualityControl`: Quality control variables for both input and output.
- `cp::CorrectionParams`: Updated correction parameters.
- `method::T`: The `FluxEstimationMethod` used for estimation.
"""
@kwdef struct FluxEstimate{T <: FluxEstimationMethod}
    estimate::NamedTuple
    qc::QualityControl
    cp::CorrectionParams
    method::T
    units::NamedTuple = NamedTuple(k => output_variables[k] for k in intersect(keys(estimate), keys(output_variables)))
end


function Base.show(io::IO, e::FluxEstimate{T}) where {T <: FluxEstimationMethod}
    print(io, "Flux Estimation Results with method $(T)\n")
    S = string.(collect(keys(e.estimate)))
    filter!(!endswith("_QC"), S)
    if T <: Union{TurbuThreshold, TurbuLaplacian}
        print(io, "\nTime-Frequency variables: ")
        for s in S
            ss = Symbol(s)
            if !endswith(s, "_TF")
                continue
            end
            print(io, s)
            if ss in keys(output_variables)
                unit = output_variables[ss]
                print(io, " ($unit)")
            end
            print(io, ", ")
        end
        if "ETA" in S
            print(io, "\nNormalized Frequency: ETA")
        end
    end
    print(io, "\nTime variables: ")
    for s in S
        ss = Symbol(s)
        if endswith(s, "_TF")
            continue
        end
        print(io, s)
        if ss in keys(output_variables)
            unit = output_variables[ss]
            if unit != NoUnits
                print(io, " ($unit)")
            end
        end
        print(io, ", ")
    end
    print(io, "\nQuality Control variables are ending with _QC")
    return nothing
end

function tofluxunits(F::AbstractArray, density::AbstractVector, fluxtype::Symbol)
    if fluxtype in (:H, :H_TF)
        F = F .* density * C_p #W/m2
    elseif fluxtype in (:LE, :LE_TF)
        F = F .* density * LAMBDA #W/m2
    elseif fluxtype in (:FC, :FC_TF)
        F = F .* density # umol/m2/s
    elseif fluxtype in (:TAUW, :TAUW_TF)
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
    mean_std_wind([u,v,w],tp::TimeParams)

Compute the mean and standard deviations `U_MEAN, U_SIGMA, V_MEAN, ...`.
"""
mean_std_wind(u, v, w, tp::TimeParams) = mean_std_wind([u, v, w], tp)
function mean_std_wind(wind_speeds, tp::TimeParams)
    length(wind_speeds) == 3 ||
        throw(error("Three wind speed signals expected, got $(length(wind_speeds))"))
    allequal(length, wind_speeds) ||
        throw(error("Got wind speed signals of different sizes"))
    u, v, w = wind_speeds
    subsampling = false
    U_MEAN = average(u, tp, subsampling)
    V_MEAN = average(v, tp, subsampling)
    W_MEAN = average(w, tp, subsampling)
    up = u - U_MEAN
    vp = v - V_MEAN
    wp = w - W_MEAN
    U_SIGMA = sqrtz.(average(up .^ 2, tp))
    V_SIGMA = sqrtz.(average(vp .^ 2, tp))
    W_SIGMA = sqrtz.(average(wp .^ 2, tp))
    sampling = 1:tp.dt:length(U_MEAN)
    U_MEAN = U_MEAN[sampling]
    V_MEAN = V_MEAN[sampling]
    W_MEAN = W_MEAN[sampling]
    return U_MEAN, V_MEAN, W_MEAN, U_SIGMA, V_SIGMA, W_SIGMA
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
function flux_scale_integral(scalo::AbstractArray{<:Real, 2}, mask::AbstractArray{<:Real, 2})
    size(mask) == size(scalo) || throw(error("Wrong size between mask and scalogram"))
    return vec(sum(scalo .* mask, dims = 2))
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
    C = [(1, i + 1) for i in 1:length(s)]
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
        tauw::AbstractArray{<:Real, 2},
        method::TurbuThreshold,
        mask_error::Union{Nothing, AbstractArray{Bool}} = nothing,
    )
    (; tr_tau) = method
    mask = tauw .> tr_tau
    mask[:, end] .= false # always remove zero frequency band
    if !isnothing(mask_error)
        mask = mask .&& .!mask_error
    end
    return (; TAUW_TF_M = mask)
end

function turbulence_mask(
        tauw::AbstractArray{<:Real, 2},
        WS::AbstractArray{<:Real, 1},
        aux::AuxVars,
        method::TurbuLaplacian,
        mask_error::Union{Nothing, AbstractArray{Bool}} = nothing,
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
    mask[:, freq_peaks .== 0] .= false

    adj_mat = grid_adj_mat(S, mask) # 9-point grid adjacency matrix with removed vertices from mask
    weights_mat = adj_mat # Using the adjacency matrix as the weight matrix amounts to compute a normal laplacian

    g = MyGraph(adj_mat, weights_mat)
    L = laplacian_matrix(g)
    dtau = reshape(L * ltauw_v, S) # the laplacian is zero where mask is false
    ltauw_v = reshape(ltauw_v, S)

    detected = reshape(dtau .> tr_dtau, S) # Look at (t,eta) points with important minimas
    if !isnothing(mask_error)
        detected[mask_error] .= false # remove points with convolution errors
    end
    detected[(eta .> 1)] .= false # remove points above eta = 0

    if count(detected) == 0
        sg = eta[:, end]
        mask_advec = mask # Get the mask removing the advection + removing the mean value
    else
        #itp = _interpolate_eta(t[detected[:]], eta[detected[:]], λ,d) # Old way: Bspline interpolation + smoothness regularization to get interpolated value at each time t, extrapolate with constant values on the borders
        #
        ts = collect(Iterators.flatten((t[view(detected, :, i)] for i in 1:size(detected, 2))))
        itp = _locally_weighted_regression(ts, eta[detected], span)
        sg = itp.(t) #Time-Varying spectral gap
        mask_advec = (eta .> sg) .&& mask # Get the mask removing the advection + removing the mean value
    end

    mask_lowcoeff = (ltauw_v .> log10(tr_tau)) .&& mask_advec
    return (;
        TAUW_TF_M = mask_lowcoeff,
        SG = sg,
        TAUW_TF_MADVEC = mask_advec,
        DTAUW_TF = dtau,
    )
end

function turbulence_mask(
        tauw::AbstractArray{<:Real, 2},
        WS::AbstractArray{<:Real, 1},
        aux::AuxVars,
        method::TurbuMorpho,
        mask_error::Union{Nothing, AbstractArray{Bool}} = nothing,
    )
    (; tr_tau, a, N, dp) = method
    (; z_d) = aux
    freq_peaks = frequency_peaks(dp)
    eta = normalized_frequency(freq_peaks, WS, z_d)
    B = tauw .> tr_tau
    B[:, end] .= false # always remove zero frequency band
    R = circle(N)
    Bo = erode(B, R)
    Bf = open_reconstruct(Bo, B, circle(3))
    mask = Bf
    if !isnothing(mask_error)
        mask = mask .&& .!mask_error
    end
    mask = smooth_mask(mask, a)
    return (; TAUW_TF_M = mask)
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

    U_MEAN, V_MEAN, W_MEAN, U_SIGMA, V_SIGMA, W_SIGMA = mean_std_wind(df[:U], df[:V], df[:W], tp)
    turbuvar[:U_SIGMA] = U_SIGMA
    turbuvar[:U_MEAN] = U_MEAN
    update_quality_control!(qc, :U_SIGMA, error_mask(tp, qc[:U]))
    update_quality_control!(qc, :U_MEAN, error_mask(tp, qc[:U]))

    turbuvar[:V_SIGMA] = V_SIGMA
    turbuvar[:V_MEAN] = V_MEAN
    update_quality_control!(qc, :V_SIGMA, error_mask(tp, qc[:V]))
    update_quality_control!(qc, :V_MEAN, error_mask(tp, qc[:V]))


    turbuvar[:W_SIGMA] = W_SIGMA
    turbuvar[:W_MEAN] = W_MEAN
    update_quality_control!(qc, :W_SIGMA, error_mask(tp, qc[:W]))
    update_quality_control!(qc, :W_MEAN, error_mask(tp, qc[:W]))
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

_C_dp(C) = Dict(k => Symbol(split(string(v), "_TF")[1], "_DSIGMA", "_TF") for (k, v) in C)
_C_dt(C) = Dict(k => Symbol(split(string(v), "_TF")[1], "_DT", "_TF") for (k, v) in C)

"""

    estimate_flux(; df::Dict{Symbol,<:AbstractArray}, aux::AuxVars, cp::CorrectionParams, method::FluxEstimationMethod)::FluxEstimate

Estimate fluxes using input data, auxiliary variables, correction and method parameters. Returns a `FluxEstimate` object containing the results.

# Arguments
- `df::Dict{Symbol,<:AbstractArray}`: A dictionary mapping `Symbol` keys to arrays of data. The keys and units must follow the conventions specified in [`mandatory_variables`](@ref) and [`gas_variables`](@ref).
- `aux::AuxVars`: A struct containing auxiliary variables, such as sampling frequency. See [`AuxVars`](@ref) for details.
- `cp::CorrectionParams`: A struct containing correction parameters. See [`CorrectionParams`](@ref) for details.
- `method::FluxEstimationMethod`: A struct specifying the parameters for the flux estimation method. See [`FluxEstimationMethod`](@ref) for details.

# Returns
- `::FluxEstimate`: An object containing the estimated fluxes and associated metadata.
"""
estimate_flux(; df, aux, cp, method) = estimate_flux(df, aux, cp, method)
estimate_flux(df, aux::AuxVars, cp::CorrectionParams, method::FluxEstimationMethod) = estimate_flux(to_dict(df), aux, cp, method)
function estimate_flux(
        df::Dict{Symbol, <:Any},
        aux::AuxVars,
        cp::CorrectionParams,
        method::ReynoldsEstimation,
    )
    (; tp, tp_aux) = method
    tp.dt == tp_aux.dt || throw(
        error(
            "Different time sampling given: tp.dt = $(tp.dt) against tp_aux.dt = $(tp_aux.dt)",
        ),
    )
    (; dt) = tp
    df = deepcopy(df)
    check_variables(df)
    var_names = get_var_names(df)
    work_dim = length(df[:TIMESTAMP])
    time_sampling = 1:dt:work_dim
    estimate = Dict() # Prepare Output
    estimate[:TIMESTAMP] = df[:TIMESTAMP][time_sampling]
    df, qc = apply_correction!(df, cp, aux)

    auxvars = estimate_auxvar(df, tp_aux, qc)
    merge!(estimate, auxvars)
    turbuvar = estimate_turbuvar(df, tp, qc)
    merge!(estimate, turbuvar)

    C = Dict((:W, :TA) => :H)
    if :CO2 in var_names
        C[(:W, :CO2)] = :FC
    end
    if :H2O in var_names
        C[(:W, :H2O)] = :LE
    end
    L = Dict(k => df[k] for k in unique(Iterators.flatten(collect(keys(C)))))
    cross_co = cross_correlation_rey(L, C, tp)
    for v in keys(cross_co)
        cross_co[v] = tofluxunits(cross_co[v], estimate[:RHO], v)
    end
    merge!(estimate, cross_co)
    for (c, v) in C
        n, m = c
        update_quality_control!(qc, v, error_mask(tp, qc[n] .|| qc[m]) .|| qc[:RHO])
    end
    estimate = to_nt(merge_qc!(estimate, qc))
    return FluxEstimate(; estimate, qc, cp, method)
end

function estimate_flux(
        df::Dict{Symbol, <:Any},
        aux::AuxVars,
        cp::CorrectionParams,
        method::Union{TurbuThreshold, TurbuLaplacian, TurbuMorpho},
    )
    (; dp, tp_aux, tp_tau) = method
    (; tp) = dp
    df = deepcopy(df)
    check_variables(df)
    var_names = get_var_names(df)
    work_dim = length(df[:TIMESTAMP])
    dp.tp.dt == tp_aux.dt || throw(
        error(
            "Different time sampling given: tp.dt = $(dp.tp.dt) against tp_aux.dt = $(tp_aux.dt)",
        ),
    )
    dp.tp.dt == tp_tau.dt || throw(
        error(
            "Different time sampling given: tp.dt = $(dp.tp.dt) against tp_tau.dt = $(tp_tau.dt)",
        ),
    )
    dp_tau = DecompParams(dp.sp, tp_tau)
    time_sampling = 1:tp_aux.dt:work_dim
    estimate = Dict()
    estimate[:TIMESTAMP] = df[:TIMESTAMP][time_sampling]
    df, qc = apply_correction!(df, cp, aux)
    auxvars = estimate_auxvar(df, tp_aux, qc)
    merge!(estimate, auxvars)
    turbuvar = estimate_turbuvar(df, tp_aux, qc)
    merge!(estimate, turbuvar)
    C_turbu = Dict((:W, :W) => :WW_TF, (:W, :U) => :UW_TF, (:W, :V) => :VW_TF)
    C_flux = Dict((:W, :TA) => :H_TF)
    if :CO2 in var_names
        C_flux[(:W, :CO2)] = :FC_TF
    end
    if :H2O in var_names
        C_flux[(:W, :H2O)] = :LE_TF
    end
    L_turbu = Dict(k => df[k] for k in unique(Iterators.flatten(collect(keys(C_turbu)))))
    L_flux = Dict(k => df[k] for k in unique(Iterators.flatten(collect(keys(C_flux)))))
    @info "Cross-Scalogram computation for $(union(values(C_turbu), values(C_flux)))"
    scalo_turbu = cross_scalogram(L_turbu, C_turbu, dp_tau)
    scalo_flux = cross_scalogram(L_flux, C_flux, dp)
    scalo = merge(scalo_turbu, scalo_flux)
    for c in keys(scalo)
        scalo[c] = tofluxunits(scalo[c], estimate[:RHO], c)
    end
    scalo[:TAUW_TF] = _tauw(scalo[:WW_TF], scalo[:UW_TF], scalo[:VW_TF])

    scalo_flux_qc = qc_cross_scalogram(qc, C_flux, dp)
    scalo_turbu_qc = qc_cross_scalogram(qc, C_turbu, dp_tau)
    scalo_qc = merge(scalo_flux_qc, scalo_turbu_qc)
    for c in values(C_flux)
        qc[Symbol(c, :_QC)] = sparse(abs.(scalo_qc[Symbol(c, :_QC)]) .> 1.0e-6) .|| qc[:RHO]
    end
    for c in values(C_turbu)
        qc[Symbol(c, :_QC)] = sparse(abs.(scalo_qc[Symbol(c, :_QC)]) .> 1.0e-6)
    end
    qc[:TAUW_TF_QC] = qc[:WW_TF_QC] .|| qc[:UW_TF_QC] .|| qc[:VW_TF_QC]

    if method isa TurbuLaplacian
        tm =
            turbulence_mask(scalo[:TAUW_TF], estimate[:WS], aux, method)
        estimate[:SG] = tm.SG
        estimate[:DTAUW_TF] = tm.DTAUW_TF
        estimate[:TAUW_TF_M] = tm.TAUW_TF_M
        estimate[:TAUW_TF_MADVEC] = tm.TAUW_TF_MADVEC
    elseif method isa TurbuMorpho
        tm = turbulence_mask(scalo[:TAUW_TF], estimate[:WS], aux, method)
        estimate[:TAUW_TF_M] = tm.TAUW_TF_M
    elseif method isa TurbuThreshold
        tm = turbulence_mask(scalo[:TAUW_TF], method)
        estimate[:TAUW_TF_M] = tm.TAUW_TF_M
    end

    # Scale Integral
    for c in union(values(C_flux), values(C_turbu))
        var = Symbol(split(string(c), "_TF")[1])
        estimate[var] = flux_scale_integral(scalo[c], estimate[:TAUW_TF_M])
        mask = sparse(flux_scale_integral(qc[Symbol(c, :_QC)], estimate[:TAUW_TF_M]) .> 1.0e-6)
        update_quality_control!(qc, var, mask)
    end
    estimate[:TAUW] = flux_scale_integral(scalo[:TAUW_TF], estimate[:TAUW_TF_M])
    mask = sparse(flux_scale_integral(qc[:TAUW_TF_QC], estimate[:TAUW_TF_M]) .> 1.0e-6)
    update_quality_control!(qc, :TAUW, mask)

    if method.sensitivity
        for (L, C, dp) in [(L_flux, C_flux, dp), (L_turbu, C_turbu, dp_tau)]
            C_dp = _C_dp(C)
            C_dt = _C_dt(C)
            dp_scalo = dp_cross_scalogram(L, _C_dp(C), dp)
            dt_scalo = dt_cross_scalogram(L, _C_dt(C), dp)
            for c in keys(dp_scalo)
                dp_scalo[c] = tofluxunits(dp_scalo[c], estimate[:RHO], c)
            end
            for c in keys(dt_scalo)
                dt_scalo[c] = tofluxunits(dt_scalo[c], estimate[:RHO], c)
            end
            merge!(scalo, dp_scalo, dt_scalo)
            for c in union(values(C_dp), values(C_dt))
                var = Symbol(split(string(c), "_TF")[1])
                estimate[var] = flux_scale_integral(scalo[c], estimate[:TAUW_TF_M])
            end
        end
    end
    merge!(estimate, scalo)
    freq_peaks = frequency_peaks(dp)
    estimate[:ETA] = normalized_frequency(freq_peaks, estimate[:WS], aux.z_d)
    estimate = to_nt(merge_qc!(estimate, qc))
    return FluxEstimate(; estimate, qc, cp, method)
end

function method_range_and_timestep(method::Union{TurbuThreshold, TurbuLaplacian, TurbuMorpho}, work_dim::Integer, fs::Real, f::Real; rounding = :Hour)
    mask = converror_mask(method.dp, work_dim; subsampling = false)
    sampling = 1:method.dp.tp.dt:work_dim
    K = Union{Nothing, Int}[ findfirst(view(.!mask, :, i)) for i in axes(mask, 2) ]
    K[isnothing.(K)] .= div(work_dim, 2)
    freq_peaks = fs * frequency_peaks(method.dp)
    i = findmin(abs, freq_peaks .- f)[2]
    if rounding == :Hour
        N = round(Int, K[i] / (fs * 3600)) * fs * 3600
    elseif isnothing(rounding)
        N = K[i]
    end
    range = trues(work_dim)
    range[1:N] .= false
    range[(end - N + 1):end] .= false
    timestep = work_dim - 2 * N
    return range[sampling], timestep
end

function method_timestep(method::ReynoldsEstimation, work_dim::Int)
    mask = converror_mask(method.tp, work_dim; subsampling = false)
    K = findfirst(.!mask)
    return work_dim - 2 * K
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
